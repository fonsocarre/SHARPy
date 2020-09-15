"""
Linear beam model class

S. Maraniello, Aug 2018
N. Goizueta
"""

import numpy as np
import scipy as sc
import scipy.signal as scsig
import sharpy.linear.src.libss as libss
import sharpy.utils.algebra as algebra
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import warnings


class FlexDynamic():
    r"""
    Define class for linear state-space realisation of GEBM flexible-body
    equations from SHARPy``timestep_info`` class and with the nonlinear structural information.

    The linearised beam takes the following arguments:

    Args:
        tsinfo (sharpy.utils.datastructures.StructImeStepInfo): Structural timestep containing the modal information
        structure (sharpy.solvers.beamloader.Beam): Beam class with the structural information
        custom_settings (dict): settings for the linearised beam


    State-space models can be defined in continuous or discrete time (dt
    required). Modal projection, either on the damped or undamped modal shapes,
    is also avaiable. The rad/s array wv can be optionally passed for freq.
    response analysis

    To produce the state-space equations:

    1. Set the settings:
        a. ``modal_projection={True,False}``: determines whether to project the states
            onto modal coordinates. Projection over damped or undamped modal
            shapes can be obtained selecting:

                - ``proj_modes={'damped','undamped'}``

            while

                 - ``inout_coords={'modes','nodal'}``

             determines whether the modal state-space inputs/outputs are modal
             coords or nodal degrees-of-freedom. If ``modes`` is selected, the
             ``Kin`` and ``Kout`` gain matrices are generated to transform nodal to modal
             dofs

        b. ``dlti={True,False}``: if true, generates discrete-time system.
            The continuous to discrete transformation method is determined by::

                discr_method={ 'newmark',  # Newmark-beta
                                    'zoh',		# Zero-order hold
                                    'bilinear'} # Bilinear (Tustin) transformation

            DLTIs can be obtained directly using the Newmark-:math:`\beta` method

                ``discr_method='newmark'``
                ``newmark_damp=xx`` with ``xx<<1.0``

            for full-states descriptions (``modal_projection=False``) and modal projection
            over the undamped structural modes (``modal_projection=True`` and ``proj_modes``).
            The Zero-order holder and bilinear methods, instead, work in all
            descriptions, but require the continuous state-space equations.

    2. Generate an instance of the beam

    2. Run ``self.assemble()``. The method accepts an additional parameter, ``Nmodes``,
    which allows using a lower number of modes than specified in ``self.Nmodes``

    Examples:

        >>> beam_settings = {'modal_projection': True,
        >>>             'inout_coords': 'modes',
        >>>             'discrete_time': False,
        >>>             'proj_modes': 'undamped',
        >>>             'use_euler': True}
        >>>
        >>> beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure, custom_settings=beam_settings)
        >>>
        >>> beam.assemble()

    Notes:

        * Modal projection will automatically select between damped/undamped modes shapes, based on the data available
          from tsinfo.

        * If the full system matrices are available, use the modal_sol methods to override mode-shapes and eigenvectors


    """

    def __init__(self, tsinfo, structure=None, custom_settings=dict()):
        # Extract settings
        self.settings = custom_settings

        ### extract timestep_info modal results
        # unavailable attrs will be None
        self.freq_natural = tsinfo.modal.get('freq_natural')
        self.freq_damp = tsinfo.modal.get('freq_damped')

        self.damping = tsinfo.modal.get('damping')

        self.eigs = tsinfo.modal.get('eigenvalues')
        self.U = tsinfo.modal.get('eigenvectors')
        self.V = tsinfo.modal.get('eigenvectors_left')
        self.Kin_damp = tsinfo.modal.get('Kin_damp')  # for 'damped' modes only
        self.Ccut = tsinfo.modal.get('Ccut')  # for 'undamp' modes only

        self.Mstr = tsinfo.modal.get('M')
        self.Cstr = tsinfo.modal.get('C')
        self.Kstr = tsinfo.modal.get('K')

        ### set other flags
        self.modal = self.settings['modal_projection']
        self.inout_coords = self.settings['inout_coords']
        self.dlti = self.settings['discrete_time']

        if self.dlti:
            self.dt = self.settings['dt']
        else:
            self.dt = None

        self.Nmodes = self.settings['num_modes']
        self._num_modes = None
        self.num_modes = self.settings['num_modes']
        self.num_dof = self.U.shape[0]
        if self.V is not None:
            self.num_dof = self.num_dof // 2

        self.proj_modes = self.settings['proj_modes']
        if self.V is None:
            self.proj_modes = 'undamped'
        self.discr_method = self.settings['discr_method']
        self.newmark_damp = self.settings['newmark_damp']
        self.use_euler = self.settings['use_euler']

        ### set state-space variables
        self.SScont = None
        self.SSdisc = None
        self.Kin = None
        self.Kout = None

        # Store structure at linearisation and linearisation conditions
        self.structure = structure
        self.tsstruct0 = tsinfo
        self.Minv = None

        self.scaled_reference_matrices = dict()  # keep reference values prior to time scaling

        if self.use_euler:
            self.euler_propagation_equations(tsinfo)

        self.update_modal()

        self.U = self.sort_repeated_evecs(self.U, self.eigs)

        if self.Mstr.shape[0] == 6*(self.tsstruct0.num_node - 1):
            self.clamped = True
            self.num_dof_rig = 0
        else:
            self.clamped = False

        if self.use_euler:
            self.num_dof_rig = 9
        else:
            self.num_dof_rig = 10

        self.num_dof_flex = np.sum(structure.vdof >= 0)*6

        self.num_dof_str = self.num_dof_flex + self.num_dof_rig
        q, dq = self.reshape_struct_input()

        self.tsstruct0.q = q
        self.tsstruct0.dq = dq

        # Linearised gravity matrices
        self.Crr_grav = None
        self.Csr_grav = None
        self.Krs_grav = None
        self.Kss_grav = None

    def reshape_struct_input(self):
        """ Reshape structural input in a column vector """

        structure = self.structure  # self.data.aero.beam
        tsdata = self.tsstruct0

        q = np.zeros(self.num_dof_str)
        dq = np.zeros(self.num_dof_str)

        jj = 0  # structural dofs index
        for node_glob in range(structure.num_node):

            ### detect bc at node (and no. of dofs)
            bc_here = structure.boundary_conditions[node_glob]
            if bc_here == 1:  # clamp
                dofs_here = 0
                continue
            elif bc_here == -1 or bc_here == 0:
                dofs_here = 6
                jj_tra = [jj, jj + 1, jj + 2]
                jj_rot = [jj + 3, jj + 4, jj + 5]

            # retrieve element and local index
            ee, node_loc = structure.node_master_elem[node_glob, :]

            # allocate
            q[jj_tra] = tsdata.pos[node_glob, :]
            q[jj_rot] = tsdata.psi[ee, node_loc]
            # update
            jj += dofs_here

        # allocate FoR A quantities
        if self.use_euler:
            q[-9:-3] = tsdata.for_vel
            q[-3:] = algebra.quat2euler(tsdata.quat)

            wa = tsdata.for_vel[3:]
            dq[-9:-3] = tsdata.for_acc
            T = algebra.deuler_dt(q[-3:])
            dq[-3:] = T.dot(wa)

        else:
            q[-10:-4] = tsdata.for_vel
            q[-4:] = tsdata.quat

            wa = tsdata.for_vel[3:]
            dq[-10:-4] = tsdata.for_acc
            dq[-4] = -0.5 * np.dot(wa, tsdata.quat[1:])

        return q, dq

    @property
    def num_modes(self):
        return self._num_modes

    @num_modes.setter
    def num_modes(self, value):
        self.update_truncated_modes(value)
        self._num_modes = value

    @property
    def num_flex_dof(self):
        return np.sum(self.structure.vdof >= 0) * 6

    @property
    def num_rig_dof(self):
        return self.Mstr.shape[0] - self.num_flex_dof

    def sort_repeated_evecs(self, evecs, evals):
        num_rbm = np.sum(evals.__abs__() == 0.)
        num_dof = evecs.shape[0]

        evecs_sorted = evecs.copy()

        if num_rbm != 0:
            for i in range(num_rbm):
                index_mode = np.argmax(evecs[:, i].__abs__()) - num_dof + num_rbm
                evecs_sorted[:, index_mode] = evecs[:, i]

        return evecs_sorted

    def euler_propagation_equations(self, tsstr):
        """
        Introduce the linearised Euler propagation equations that relate the body fixed angular velocities to the Earth
        fixed Euler angles.

        This method will remove the quaternion propagation equations created by SHARPy; the resulting
        system will have 9 rigid degrees of freedom.


        Args:
            tsstr:

        Returns:

        """

        # Verify the rigid body modes are used
        num_node = tsstr.num_node
        num_flex_dof = 6*(num_node-1)

        euler = algebra.quat2euler(tsstr.quat)
        tsstr.euler = euler

        if self.Mstr.shape[0] == num_flex_dof + 10:

            # Erase quaternion equations
            self.Cstr[-4:, :] = 0

            self.Mstr = self.Mstr[:-1, :-1]
            self.Cstr = self.Cstr[:-1, :-1]
            self.Kstr = self.Kstr[:-1, :-1]

            for_rot = tsstr.for_vel[3:]
            Crr = np.zeros((9, 9))

            # Euler angle propagation equations
            Crr[-3:, -6:-3] = -algebra.deuler_dt(tsstr.euler)
            Crr[-3:, -3:] = -algebra.der_Teuler_by_w(tsstr.euler, for_rot)

            self.Cstr[-9:, -9:] += Crr

        else:
            warnings.warn('Euler parametrisation not implemented - Either rigid body modes are not being used or this '
                          'method has already been called.')


    @property
    def num_dof(self):
        self.num_dof = self.Mstr.shape[0]
        # Previously beam.U.shape[0]
        return self._num_dof

    @num_dof.setter
    def num_dof(self, value):
        self._num_dof = value


    def linearise_gravity_forces(self, tsstr=None):
        r"""
        Linearises gravity forces and includes the resulting terms in the C and K matrices. The method takes the
        linearisation condition (optional argument), linearises and updates:

            * Stiffness matrix

            * Damping matrix

            * Modal damping matrix

        The method works for both the quaternion and euler angle orientation parametrisation.

        Args:
            tsstr (sharpy.utils.datastructures.StructTimeStepInfo): Structural timestep at the linearisation point

        Notes:

            The gravity forces are linearised to express them in terms of the beam formulation input variables:

                * Nodal forces: :math:`\delta \mathbf{f}_A`

                * Nodal moments: :math:`\delta(T^T \mathbf{m}_B)`

                * Total forces (rigid body equations): :math:`\delta \mathbf{F}_A`

                * Total moments (rigid body equations): :math:`\delta \mathbf{M}_A`

            Gravity forces are naturally expressed in ``G`` (inertial) frame

            .. math:: \mathbf{f}_{G,0} = \mathbf{M\,g}

            where the :math:`\mathbf{M}` is the tangent mass matrix obtained at the linearisation reference.

            To obtain the gravity forces expressed in A frame we make use of the projection matrices

            .. math:: \mathbf{f}_A = C^{AG}(\boldsymbol{\chi}) \mathbf{f}_{G,0}

            that projects a vector in the inertial frame ``G`` onto the body attached frame ``A``.

            The projection of a vector can then be linearised as

            .. math::
                \delta \mathbf{f}_A = C^{AG} \delta \mathbf{f}_{G,0}
                + \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG} \mathbf{f}_{G,0}) \delta\boldsymbol{\chi}.

            * Nodal forces:

                The linearisation of the gravity forces acting at each node is simply

                .. math::
                    \delta \mathbf{f}_A =
                    + \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG} \mathbf{f}_{G,0}) \delta\boldsymbol{\chi}

                where it is assumed that :math:`\delta\mathbf{f}_G = 0`.

            * Nodal moments:

                The gravity moments can be expressed in the local node frame of reference ``B`` by

                .. math:: \mathbf{m}_B = \tilde{X}_{B,CG}C^{BA}(\Psi)C^{AG}(\boldsymbol{\chi})\mathbf{f}_{G,0}

                The linearisation is given by:

                .. math::
                    \delta \mathbf{m}_B = \tilde{X}_{B,CG}
                    \left(\frac{\partial}{\partial\Psi}(C^{BA}\mathbf{f}_{A,0})\delta\Psi +
                    C^{BA}\frac{\partial}{\partial\boldsymbol{\chi}}(C^{AG}\mathbf{f}_{G,0})\delta\boldsymbol{\chi}\right)

                However, recall that the input moments are defined in tangential space
                :math:`\delta(T^\top\mathbf{m}_B)` whose linearised expression is

                .. math:: \delta(T^T(\Psi) \mathbf{m}_B) = T_0^T \delta \mathbf{m}_B +
                    \frac{\partial}{\partial \Psi}(T^T \mathbf{m}_{B,0})\delta\Psi

                where the :math:`\delta \mathbf{m}_B` term has been defined above.

            * Total forces:

                The total forces include the contribution from all flexible degrees of freedom as well as the gravity
                forces arising from the mass at the clamped node

                .. math:: \mathbf{F}_A = \sum_n \mathbf{f}_A + \mathbf{f}_{A,clamped}

                which becomes

                .. math:: \delta \mathbf{F}_A = \sum_n \delta \mathbf{f}_A +
                    \frac{\partial}{\partial\boldsymbol{\chi}}\left(C^{AG}\mathbf{f}_{G,clamped}\right)
                    \delta\boldsymbol{\chi}.

            * Total moments:

                The total moments, as opposed to the nodal moments, are expressed in A frame and again require the
                addition of the moments from the flexible structural nodes as well as the ones from the clamped node
                itself.

                .. math:: \mathbf{M}_A = \sum_n \tilde{X}_{A,n}^{CG} C^{AG} \mathbf{f}_{n,G}
                    + \tilde{X}_{A,clamped}C^{AG}\mathbf{f}_{G, clamped}

                where :math:`X_{A,n}^{CG} = R_{A,n} + C^{AB}(\Psi)X_{B,n}^{CG}`. Its linearised form is

                .. math:: \delta X_{A,n}^{CG} = \delta R_{A,n}
                    + \frac{\partial}{\partial \Psi}(C^{AB} X_{B,CG})\delta\Psi

                Therefore, the overall linearisation of the total moment is defined as

                .. math:: \delta \mathbf{M}_A =
                    \tilde{X}_{A,total}^{CG} \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG}\mathbf{F}_{G, total})
                    \delta \boldsymbol{\chi}
                    -\sum_n \tilde{C}^{AG}\mathbf{f}_{G,0} \delta X_{A,n}^{CG}

                where :math:`X_{A, total}` is the centre of gravity of the entire system expressed in ``A`` frame and
                :math:`\mathbf{F}_{G, total}` are the gravity forces of the overall system in ``G`` frame, including the
                contributions from the clamped node.


            The linearisation introduces damping and stiffening terms since the :math:`\delta\boldsymbol{\chi}` and
            :math:`\delta\boldsymbol{\Psi}` terms are found in the damping and stiffness matrices respectively.

            Therefore, the beam matrices need updating to account for these terms:

                * Terms from the linearisation of the nodal moments will be assembled in the rows corresponding to
                  moment equations and columns corresponding to the cartesian rotation vector

                    .. math:: K_{ss}^{m,\Psi} \leftarrow -T_0^T \tilde{X}_{B,CG}
                        \frac{\partial}{\partial\Psi}(C^{BA}\mathbf{f}_{A,0})
                        -\frac{\partial}{\partial \Psi}(T^T \mathbf{m}_{B,0})

                * Terms from the linearisation of the translation forces with respect to the orientation are assembled
                  in the damping matrix, the rows corresponding to translational forces and columns to orientation
                  degrees of freedom

                    .. math:: C_{sr}^{f,\boldsymbol{\chi}} \leftarrow -
                        \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG} \mathbf{f}_{G,0})

                * Terms from the linearisation of the moments with respect to the orientation are assembled in the
                  damping matrix, with the rows correspondant to the moments and the columns to the orientation degrees
                  of freedom

                    .. math:: C_{sr}^{m,\boldsymbol{\chi}} \leftarrow -
                        T_0^T\tilde{X}_{B,CG}C^{BA}\frac{\partial}{\partial\boldsymbol{\chi}}(C^{AG}\mathbf{f}_{G,0})

                * Terms from the linearisation of the total forces with respect to the orientation correspond to the
                  rigid body equations in the damping matrix, the rows to the translational forces and columns to the
                  orientation

                    .. math:: C_{rr}^{F,\boldsymbol{\chi}} \leftarrow
                        - \sum_n \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG} \mathbf{f}_{G,0})

                * Terms from the linearisation of the total moments with respect to the orientation correspond to the
                  rigid body equations in the damping matrix, the rows to the moments and the columns to the orientation

                    .. math:: C_{rr}^{M,\boldsymbol{\chi}} \leftarrow
                        - \sum_n\tilde{X}_{A,n}^{CG} \frac{\partial}{\partial \boldsymbol{\chi}}(C^{AG}\mathbf{f}_{G,0})

                * Terms from the linearisation of the total moments with respect to the nodal position :math:`R_A` are
                  included in the stiffness matrix, the rows corresponding to the moments in the rigid body
                  equations and the columns to the nodal position

                    .. math:: K_{rs}^{M,R} \leftarrow + \sum_n \tilde{\mathbf{f}_{A,0}}

                * Terms from the linearisation of the total moments with respect to the cartesian rotation vector are
                  included in the stiffness matrix, the rows corresponding to the moments in the rigid body equations
                  and the columns to the cartesian rotation vector

                    .. math:: K_{rs}^{M, \Psi} \leftarrow
                        + \sum_n \tilde{\mathbf{f}_{A,0}}\frac{\partial}{\partial \Psi}(C^{AB} X_{B,CG})
                    
        """

        if tsstr is None:
            tsstr = self.tsstruct0

        if self.settings['print_info']:
            try:
                cout.cout_wrap('\nLinearising gravity terms...')
            except ValueError:
                pass

        num_node = tsstr.num_node
        flex_dof = 6 * sum(self.structure.vdof >= 0)
        if self.use_euler:
            rig_dof = 9
            # This is a rotation matrix that rotates a vector from G to A
            Cag = algebra.euler2rot(tsstr.euler)
            Cga = Cag.T

            # Projection matrices - this projects the vector in G t to A
            Pag = Cga
            Pga = Cag
        else:
            rig_dof = 10
            # get projection matrix A->G
            # Cga = algebra.quat2rotation(tsstr.quat)
            # Pga = Cga.T
            # Pag = Pga.T
            Cag = algebra.quat2rotation(tsstr.quat)  # Rotation matrix FoR G rotated by quat
            Pag = Cag.T
            Pga = Pag.T

        # Mass matrix partitions for CG calculations
        Mss = self.Mstr[:flex_dof, :flex_dof]
        Mrr = self.Mstr[-rig_dof:, -rig_dof:]

        # Initialise damping and stiffness gravity terms
        Crr_grav = np.zeros((rig_dof, rig_dof))
        Csr_grav = np.zeros((flex_dof, rig_dof))
        Crr_debug = np.zeros((rig_dof, rig_dof))
        Krs_grav = np.zeros((rig_dof, flex_dof))
        Kss_grav = np.zeros((flex_dof, flex_dof))

        # Overall CG in A frame
        Xcg_A = -np.array([Mrr[2, 4], Mrr[0, 5], Mrr[1, 3]]) / Mrr[0, 0]
        Xcg_Askew = algebra.skew(Xcg_A)

        if self.settings['print_info']:
            cout.cout_wrap('\tM = %.2f kg' % Mrr[0, 0], 1)
            cout.cout_wrap('\tX_CG A -> %.2f %.2f %.2f' %(Xcg_A[0], Xcg_A[1], Xcg_A[2]), 1)

        FgravA = np.zeros(3)
        FgravG = np.zeros(3)

        for i_node in range(num_node):
            # Gravity forces at the linearisation condition (from NL SHARPy in A frame)
            fgravA = tsstr.gravity_forces[i_node, :3]
            fgravG = Pga.dot(fgravA)
            # fgravG = tsstr.gravity_forces[i_node, :3]
            mgravA = tsstr.gravity_forces[i_node, 3:]
            fgravA = Pag.dot(fgravG)
            mgravG = Pag.dot(mgravA)

            # Get nodal position - A frame
            Ra = tsstr.pos[i_node, :]

            # retrieve element and local index
            ee, node_loc = self.structure.node_master_elem[i_node, :]
            psi = tsstr.psi[ee, node_loc, :]
            Cab = algebra.crv2rotation(psi)
            Cba = Cab.T
            Cbg = Cba.dot(Pag)

            # Tangential operator for moments calculation
            Tan = algebra.crv2tan(psi)

            jj = 0  # nodal dof index
            bc_at_node = self.structure.boundary_conditions[i_node]  # Boundary conditions at the node

            if bc_at_node == 1:  # clamp (only rigid-body)
                dofs_at_node = 0
                jj_tra, jj_rot = [], []

            elif bc_at_node == -1 or bc_at_node == 0:  # (rigid+flex body)
                dofs_at_node = 6
                jj_tra = 6 * self.structure.vdof[i_node] + np.array([0, 1, 2], dtype=int)  # Translations
                jj_rot = 6 * self.structure.vdof[i_node] + np.array([3, 4, 5], dtype=int)  # Rotations
            else:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (bc_at_node, i_node))

            jj += dofs_at_node

            if bc_at_node != 1:
                # Nodal centre of gravity (in the case of additional lumped masses, else should be zero)
                Mss_indices = np.concatenate((jj_tra, jj_rot))
                Mss_node = Mss[Mss_indices,:]
                Mss_node = Mss_node[:, Mss_indices]
                Xcg_B = Cba.dot(-np.array([Mss_node[2, 4], Mss_node[0, 5], Mss_node[1, 3]]) / Mss_node[0, 0])
                Xcg_Bskew = algebra.skew(Xcg_B)

                # Nodal CG in A frame
                Xcg_A_n = Ra + Cab.dot(Xcg_B)
                Xcg_A_n_skew = algebra.skew(Xcg_A_n)

                # Nodal CG in G frame - debug
                Xcg_G_n = Pga.dot(Xcg_A_n)

                if self.settings['print_info']:
                    cout.cout_wrap("Node %2d \t-> B %.3f %.3f %.3f" %(i_node, Xcg_B[0], Xcg_B[1], Xcg_B[2]), 2)
                    cout.cout_wrap("\t\t\t-> A %.3f %.3f %.3f" %(Xcg_A_n[0], Xcg_A_n[1], Xcg_A_n[2]), 2)
                    cout.cout_wrap("\t\t\t-> G %.3f %.3f %.3f" %(Xcg_G_n[0], Xcg_G_n[1], Xcg_G_n[2]), 2)
                    cout.cout_wrap("\tNode mass:", 2)
                    cout.cout_wrap("\t\tMatrix: %.4f" % Mss_node[0, 0], 2)
                    # cout.cout_wrap("\t\tGrav: %.4f" % (np.linalg.norm(fgravG)/9.81), 2)

            if self.use_euler:
                if bc_at_node != 1:
                    # Nodal moments due to gravity -> linearisation terms wrt to delta_psi
                    Kss_grav[np.ix_(jj_rot, jj_rot)] -= Tan.dot(Xcg_Bskew.dot(algebra.der_Ccrv_by_v(psi, fgravA)))
                    Kss_grav[np.ix_(jj_rot, jj_rot)] -= algebra.der_TanT_by_xv(psi, Xcg_Bskew.dot(Cbg.dot(fgravG)))

                    # Nodal forces due to gravity -> linearisation terms wrt to delta_euler
                    Csr_grav[jj_tra, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, fgravG)

                    # Nodal moments due to gravity -> linearisation terms wrt to delta_euler
                    Csr_grav[jj_rot, -3:] -= Tan.dot(Xcg_Bskew.dot(Cba.dot(algebra.der_Peuler_by_v(tsstr.euler, fgravG))))

                    # Total moments -> linearisation terms wrt to delta_Ra
                    # These terms are not affected by the Euler matrix. Sign is correct (+)
                    Krs_grav[3:6, jj_tra] += algebra.skew(fgravA)

                    # Total moments -> linearisation terms wrt to delta_Psi
                    Krs_grav[3:6, jj_rot] += np.dot(algebra.skew(fgravA), algebra.der_Ccrv_by_v(psi, Xcg_B))

            else:
                if bc_at_node != 1:
                    # Nodal moments due to gravity -> linearisation terms wrt to delta_psi
                    Kss_grav[np.ix_(jj_rot, jj_rot)] -= Tan.dot(Xcg_Bskew.dot(algebra.der_Ccrv_by_v(psi, fgravA)))
                    Kss_grav[np.ix_(jj_rot, jj_rot)] -= algebra.der_TanT_by_xv(psi, Xcg_Bskew.dot(Cbg.dot(fgravG)))

                    # Total moments -> linearisation terms wrt to delta_Ra
                    # Check sign (in theory it should be +=)
                    Krs_grav[3:6, jj_tra] += algebra.skew(fgravA)

                    # Total moments -> linearisation terms wrt to delta_Psi
                    Krs_grav[3:6, jj_rot] += np.dot(algebra.skew(fgravA), algebra.der_Ccrv_by_v(psi, Xcg_B))

                    # Nodal forces due to gravity -> linearisation terms wrt to delta_euler
                    Csr_grav[jj_tra, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, fgravG) # ok
                    # Crr_grav[:3, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, fgravG)  # not ok - see below

                    # Nodal moments due to gravity -> linearisation terms wrt to delta_euler
                    Csr_grav[jj_rot, -4:] -= Tan.dot(Xcg_Bskew.dot(Cba.dot(algebra.der_CquatT_by_v(tsstr.quat, fgravG))))


            # Debugging:
            FgravA += fgravA
            FgravG += fgravG

        if self.use_euler:
            # Total gravity forces acting at the A frame
            Crr_grav[:3, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, FgravG)

            # Total moments due to gravity in A frame
            Crr_grav[3:6, -3:] -= algebra.skew(Xcg_A).dot(algebra.der_Peuler_by_v(tsstr.euler, FgravG))
        else:
            # Total gravity forces acting at the A frame
            Crr_grav[:3, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, FgravG)

            # Total moments due to gravity in A frame
            Crr_grav[3:6, -4:] -= algebra.skew(Xcg_A).dot(algebra.der_CquatT_by_v(tsstr.quat, FgravG))

        # Update matrices
        self.Kstr[:flex_dof, :flex_dof] += Kss_grav

        if self.Kstr[:flex_dof, :flex_dof].shape != self.Kstr.shape:  # If the beam is free, update rigid terms as well
            self.Cstr[-rig_dof:, -rig_dof:] += Crr_grav
            self.Cstr[:-rig_dof, -rig_dof:] += Csr_grav
            self.Kstr[flex_dof:, :flex_dof] += Krs_grav

            # Save gravity matrices for post-processing
            self.Crr_grav = Crr_grav
            self.Csr_grav = Csr_grav
            self.Krs_grav = Krs_grav
            self.Kss_grav = Kss_grav

        if self.modal:
            self.Ccut = self.U.T.dot(self.Cstr.dot(self.U))

        if self.settings['print_info']:
            cout.cout_wrap('\tUpdated the beam C, modal C and K matrices with the terms from the gravity linearisation\n')

    def linearise_applied_forces(self, tsstr=None):
        r"""
        Linearise externally applied follower forces given in the local ``B`` reference frame.

        Updates the stiffness matrix with terms arising from this linearisation.

        The linearised beam equations are expressed in the following frames of reference:

            * Nodal forces: :math:`\delta \mathbf{f}_A`

            * Nodal moments: :math:`\delta(T^T \mathbf{m}_B)`

            * Total forces (rigid body equations): :math:`\delta \mathbf{F}_A`

            * Total moments (rigid body equations): :math:`\delta \mathbf{M}_A`

        Thus, when linearising externally applied follower forces projected onto the appropriate frame

        .. math:: \boldsymbol{f}_A^{ext} = C^{AB}(\boldsymbol{\psi})\boldsymbol{f}^{ext}_B

        the following terms appear:

        .. math::
            \delta\boldsymbol{f}_A^{ext} = \frac{\partial}{\partial\boldsymbol{\psi}}
            \left(C^{AB}(\boldsymbol{\psi})\boldsymbol{f}^{ext}_{0,B}\right)\delta\boldsymbol{\psi} +
            C^{AB}_0\delta\boldsymbol f_B^{ext}

        where the :math:`\delta\boldsymbol{\psi}` is a stiffenning term that needs to be included in the stiffness
        matrix. The terms will appear in the rows relating to the translational degrees of freedom and the columns that
        correspond to the cartesian rotation vector.

        .. math::
            K_{ss}^{f,\Psi} \leftarrow -\frac{\partial}{\partial\boldsymbol{\psi}}
            \left(C^{AB}(\boldsymbol{\psi})\boldsymbol{f}^{ext}_{0,B}\right)


        Externally applied moments in the material frame :math:`\boldsymbol{m}_B^{ext}` result in the following
        linearised expression:

        .. math::
            \delta(T^\top\boldsymbol{m}_B) = \frac{\partial}{\partial\boldsymbol{\psi}}\left(
            T^\top(\boldsymbol{\psi})\boldsymbol{m}^{ext}_{0,B}\right)\delta\boldsymbol{\psi} +
            T_0^\top \delta\boldsymbol{m}_B^{ext}

        Which results in the following stiffenning term:

        .. math::
            K_{ss}^{m,\Psi} \leftarrow -\frac{\partial}{\partial\boldsymbol{\psi}}\left(
            T^\top(\boldsymbol{\psi})\boldsymbol{m}^{ext}_{0,B}\right)

        The total contribution of moments must be summed up for the rigid body equations, and include contributions
        due to externally applied forces as well as moments:

        .. math::
            \boldsymbol{M}_A^{ext} = \sum_n \tilde{\boldsymbol{R}}_A C^{AB}(\boldsymbol{\psi}) \boldsymbol{f}_B^{ext} +
            \sum C^{AB}(\boldsymbol{\psi})\boldsymbol{m}_B^{ext}

        The linearisation of this term becomes

        .. math::
            \delta\boldsymbol{M}_A^{ext} = \sum\left(-\widetilde{C^{AB}_0 \boldsymbol{f}_{0,B}^{ext}}\delta \boldsymbol{R}_A
            + \widetilde{\boldsymbol{R}}\frac{\partial}{\partial\boldsymbol{\psi}}\left(C^{AB}\boldsymbol{f}_B\right)
            \delta \boldsymbol{\psi} +
            \widetilde{\boldsymbol{R}}C^{AB}\delta\boldsymbol{f}^{ext}_B\right) +
            \sum\left(\frac{\partial}{\partial\boldsymbol{\psi}}\left(C^{AB}\boldsymbol{m}_{0,B}\right)
            \delta\boldsymbol{\psi} +
            C^AB\delta\boldsymbol{m}_B^{ext}\right)

        which gives the following stiffenning terms in the rigid-flex partition of the stiffness matrix:

        .. math:: K_{ss}^{M,R} \leftarrow +\sum\widetilde{C^{AB}_0 \boldsymbol{f}_{0,B}^{ext}}

        .. math:: K_{ss}^{M,\Psi} \leftarrow -\sum\widetilde{\boldsymbol{R}}\frac{\partial}{\partial\boldsymbol{\psi}}
            \left(C^{AB}\boldsymbol{f}_{0,B}\right)

        and

        .. math:: K_{ss}^{M,\Psi} \leftarrow  -\sum\frac{\partial}{\partial\boldsymbol{\psi}}
            \left(C^{AB}\boldsymbol{m}_{0,B}\right).


        Args:
            tsstr (sharpy.utils.datastructures.StructTimeStepInfo): Linearisation time step.
        """
        if tsstr is None:
            tsstr = self.tsstruct0

        # TODO: Future feature: gains for externally applied forces (i.e. thrust inputs)

        num_node = tsstr.num_node
        flex_dof = 6 * sum(self.structure.vdof >= 0)
        if self.use_euler:
            rig_dof = 9
            # This is a rotation matrix that rotates a vector from G to A
            Cag = algebra.euler2rot(tsstr.euler)
            Cga = Cag.T

            # Projection matrices - this projects the vector in G t to A
            Pag = Cga
            Pga = Cag
        else:
            rig_dof = 10

        stiff_flex = np.zeros((flex_dof, flex_dof), dtype=float)  # flex-flex partition of K
        stiff_rig = np.zeros((rig_dof, flex_dof), dtype=float)  # rig-flex partition of K

        for i_node in range(num_node):
            fext_b = self.structure.steady_app_forces[i_node, :3]
            mext_b = self.structure.steady_app_forces[i_node, 3:]

            # retrieve element and local index
            ee, node_loc = self.structure.node_master_elem[i_node, :]
            psi = tsstr.psi[ee, node_loc, :]
            Cab = algebra.crv2rotation(psi)
            Cba = Cab.T

            # Tangential operator for moments calculation
            Tan = algebra.crv2tan(psi)

            # Get nodal position - in A frame
            Ra = tsstr.pos[i_node, :]

            jj = 0  # nodal dof index
            bc_at_node = self.structure.boundary_conditions[i_node]  # Boundary conditions at the node

            if bc_at_node == 1:  # clamp (only rigid-body)
                dofs_at_node = 0
                jj_tra, jj_rot = [], []

            elif bc_at_node == -1 or bc_at_node == 0:  # (rigid+flex body)
                dofs_at_node = 6
                jj_tra = 6 * self.structure.vdof[i_node] + np.array([0, 1, 2], dtype=int)  # Translations
                jj_rot = 6 * self.structure.vdof[i_node] + np.array([3, 4, 5], dtype=int)  # Rotations
            else:
                raise NameError('Invalid boundary condition ({}) at node {}'.format(bc_at_node, i_node))

            jj += dofs_at_node

            if bc_at_node != 1:
                # Externally applied follower forces
                stiff_flex[np.ix_(jj_tra, jj_rot)] -= algebra.der_Ccrv_by_v(psi, fext_b)
                stiff_rig[:3, jj_rot] -= algebra.der_Ccrv_by_v(psi, fext_b)  # Rigid body contribution

                # Externally applied moments in B frame
                stiff_flex[np.ix_(jj_rot, jj_rot)] -= algebra.der_TanT_by_xv(psi, mext_b)

                # Total moments
                # force contribution
                stiff_rig[3:6, jj_tra] += algebra.skew(Cab.dot(fext_b))  # delta Ra term
                stiff_rig[3:6, jj_rot] -= algebra.skew(Ra).dot(algebra.der_Ccrv_by_v(psi, fext_b))  # delta psi term

                # moment contribution
                stiff_rig[3:6, jj_rot] -= algebra.der_Ccrv_by_v(psi, mext_b)

            if bc_at_node == 1:
                # forces applied at the A-frame (clamped node) need special attention since the
                # node has an associated CRV to it's master element which may not be zero.
                # forces applied at this node only appear in the rigid-body equations
                try:
                    closest_node = self.structure.connectivities[ee, node_loc + 2]
                except IndexError:  # node is not in the first position
                    try:
                        closest_node = self.structure.connectivities[ee, node_loc + 1]
                    except IndexError:  # node is the midpoint
                        closest_node = self.structure.connectivities[ee, node_loc - 1]

                # indices of the node whos CRV applies to the clamped node
                jj_rot = 6 * self.structure.vdof[closest_node] + np.array([3, 4, 5], dtype=int)

                stiff_rig[:3, jj_rot] -= algebra.der_Ccrv_by_v(psi, fext_b)  # Rigid body contribution
                # Total moments
                # force contribution
                stiff_rig[3:6, jj_rot] += algebra.skew(Cab.dot(fext_b))  # delta Ra term
                stiff_rig[3:6, jj_rot] -= algebra.skew(Ra).dot(algebra.der_Ccrv_by_v(psi, fext_b))  # delta psi term

                # moment contribution
                stiff_rig[3:6, jj_rot] -= algebra.der_Ccrv_by_v(psi, mext_b)

        self.Kstr[:flex_dof, :flex_dof] += stiff_flex

        if self.Kstr[:flex_dof, :flex_dof].shape != self.Kstr.shape:  # free flying structure
            self.Kstr[-rig_dof:, :flex_dof] += stiff_rig

    def assemble(self, Nmodes=None):
        r"""
        Assemble state-space model

        Several assembly options are available:

        1. Discrete-time, Newmark-:math:`\beta`:
            * Modal projection onto undamped modes. It uses the modal projection such
              that the generalised coordinates :math:`\eta` are transformed into modal space by

                    .. math:: \mathbf{\eta} = \mathbf{\Phi\,q}

                where :math:`\mathbf{\Phi}` are the first ``Nmodes`` right eigenvectors.
                Therefore, the equation of motion can be re-written such that the modes normalise the mass matrix to
                become the identity matrix.

                    .. math:: \mathbf{I_{Nmodes}}\mathbf{\ddot{q}} + \mathbf{\Lambda_{Nmodes}\,q} = 0

                The system is then assembled in Newmark-:math:`\beta` form as detailed in :func:`newmark_ss`

            * Full size system assembly. No modifications are made to the mass, damping or stiffness matrices and the
              system is directly assembled by :func:`newmark_ss`.

        2. Continuous time state-space


        Args:
            Nmodes (int): number of modes to retain
        """

        ### checks
        assert self.inout_coords in ['modes', 'nodes'], \
            'inout_coords=%s not implemented!' % self.inout_coords

        # cond_mass_matrix = np.linalg.cond(self.Mstr)
        # if np.log10(cond_mass_matrix) >= 10.:
        #     warnings.warn('Mass matrix is poorly conditioned (Cond = 10^%f). Inverse may not be correct.'
        #                   % np.log10(cond_mass_matrix), 3)
        # else:
        #     cout.cout_wrap('Mass matrix condition = %e' % cond_mass_matrix)

        dlti = self.dlti
        modal = self.modal
        num_dof = self.num_dof
        if Nmodes is None or Nmodes >= self.num_modes:
            Nmodes = self.num_modes
        # else:
        #     # Modal truncation
        #     self.update_truncated_modes(Nmodes)

        if dlti:  # ---------------------------------- assemble discrete time

            if self.discr_method in ['zoh', 'bilinear']:
                # assemble continuous-time
                self.dlti = False
                self.assemble(Nmodes)
                # convert into discrete
                self.dlti = True
                self.cont2disc()

            elif self.discr_method == 'newmark':

                if modal:  # Modal projection
                    if self.proj_modes == 'undamped':
                        Phi = self.U[:, :Nmodes]

                        if self.Ccut is None:
                            # Ccut = np.zeros((Nmodes, Nmodes))
                            Ccut = np.dot(Phi.T, np.dot(self.Cstr, Phi))
                        else:
                            Ccut = np.dot(Phi.T, np.dot(self.Cstr, Phi))

                        # Ass, Bss, Css, Dss = newmark_ss(
                        #     np.eye(Nmodes),
                        #     Ccut,
                        #     np.diag(self.freq_natural[:Nmodes] ** 2),
                        #     self.dt,
                        #     self.newmark_damp)
                        Ass, Bss, Css, Dss = newmark_ss(
                            # Phi.T.dot(self.Mstr.dot(Phi)),
                            np.linalg.inv(np.dot(self.U[:, :Nmodes].T, np.dot(self.Mstr, self.U[:, :Nmodes]))),
                            # np.eye(Nmodes),
                            Ccut,
                            np.dot(self.U[:, :Nmodes].T, np.dot(self.Kstr, self.U[:, :Nmodes])),
                            # Phi.T.dot(self.Kstr.dot(Phi)),
                            # np.diag(self.freq_natural[:Nmodes]**2),
                            self.dt,
                            self.newmark_damp)
                        self.Kin = Phi.T
                        self.Kout = sc.linalg.block_diag(*[Phi, Phi])
                    else:
                        raise NameError(
                            'Newmark-beta discretisation not available ' \
                            'for projection on damped eigenvectors')

                    # build state-space model
                    self.SSdisc = libss.ss(Ass, Bss, Css, Dss, dt=self.dt)
                    if self.inout_coords == 'nodes':
                        self.SSdisc = libss.addGain(self.SSdisc, self.Kin, 'in')
                        self.SSdisc = libss.addGain(self.SSdisc, self.Kout, 'out')
                        self.Kin, self.Kout = None, None


                else:  # Full system
                    self.Minv = np.linalg.inv(self.Mstr)

                    Ass, Bss, Css, Dss = newmark_ss(
                        self.Minv, self.Cstr, self.Kstr,
                        self.dt, self.newmark_damp)
                    self.Kin = None
                    self.Kout = None
                    self.SSdisc = libss.ss(Ass, Bss, Css, Dss, dt=self.dt)

            else:
                raise NameError(
                    'Discretisation method %s not available' % self.discr_method)


        else:  # -------------------------------- assemble continuous time

            if modal:  # Modal projection

                Ass = np.zeros((2 * Nmodes, 2 * Nmodes))
                Css = np.eye(2 * Nmodes)
                iivec = np.arange(Nmodes, dtype=int)

                if self.proj_modes == 'undamped':
                    Phi = self.U[:, :Nmodes]
                    Ass[iivec, Nmodes + iivec] = 1.
                    # Ass[Nmodes:, :Nmodes] = -np.diag(self.freq_natural[:Nmodes] ** 2)
                    Ass[Nmodes:, :Nmodes] = -self.U.T.dot(self.Kstr.dot(self.U))
                    if self.Ccut is not None:
                        Ass[Nmodes:, Nmodes:] = -self.Ccut[:Nmodes, :Nmodes]
                    Bss = np.zeros((2 * Nmodes, Nmodes))
                    Dss = np.zeros((2 * Nmodes, Nmodes))
                    Bss[Nmodes + iivec, iivec] = 1.
                    self.Kin = Phi.T
                    self.Kout = sc.linalg.block_diag(*(Phi, Phi))
                else:  # damped mode shapes
                    # The algorithm assumes that for each couple of complex conj
                    # eigenvalues, only one eigenvalue (and the eigenvectors
                    # associated to it) is include in self.eigs.
                    eigs = self.eigs[:Nmodes]
                    U = self.U[:, :Nmodes]
                    V = self.V[:, :Nmodes]
                    Ass[iivec, iivec] = eigs.real
                    Ass[iivec, Nmodes + iivec] = -eigs.imag
                    Ass[Nmodes + iivec, iivec] = eigs.imag
                    Ass[Nmodes + iivec, Nmodes + iivec] = eigs.real
                    Bss = np.eye(2 * Nmodes)
                    Dss = np.zeros((2 * Nmodes, 2 * Nmodes))
                    self.Kin = np.block(
                        [[self.Kin_damp[iivec, :].real],
                         [self.Kin_damp[iivec, :].imag]])
                    self.Kout = np.block([2. * U.real, (-2.) * U.imag])

                # build state-space model
                self.SScont = libss.ss(Ass, Bss, Css, Dss)
                if self.inout_coords == 'nodes':
                    self.SScont = libss.addGain(self.SScont, self.Kin, 'in')
                    self.SScont = libss.addGain(self.SScont, self.Kout, 'out')
                    self.Kin, self.Kout = None, None

            else:  # Full system
                if self.Mstr is None:
                    raise NameError('Full-states matrices not available')
                Mstr, Cstr, Kstr = self.Mstr, self.Cstr, self.Kstr

                Ass = np.zeros((2 * num_dof, 2 * num_dof))
                Bss = np.zeros((2 * num_dof, num_dof))
                Css = np.eye(2 * num_dof)
                Dss = np.zeros((2 * num_dof, num_dof))
                Minv_neg = -np.linalg.inv(self.Mstr)
                Ass[range(num_dof), range(num_dof, 2 * num_dof)] = 1.
                Ass[num_dof:, :num_dof] = np.dot(Minv_neg, Kstr)
                Ass[num_dof:, num_dof:] = np.dot(Minv_neg, Cstr)
                Bss[num_dof:, :] = -Minv_neg
                self.Kin = None
                self.Kout = None
                self.SScont = libss.ss(Ass, Bss, Css, Dss)

    def freqresp(self, wv=None, bode=True):
        """
        Computes the frequency response of the current state-space model. If
        ``self.modal=True``, the in/out are determined according to ``self.inout_coords``
        """

        assert wv is not None, 'Frequency range not provided.'

        if self.dlti:
            self.Ydisc = libss.freqresp(self.SSdisc, wv, dlti=self.dlti)
            if bode:
                self.Ydisc_abs = np.abs(self.Ydisc)
                self.Ydisc_ph = np.angle(self.Ydisc, deg=True)
        else:
            self.Ycont = libss.freqresp(self.SScont, wv, dlti=self.dlti)
            if bode:
                self.Ycont_abs = np.abs(self.Ycont)
                self.Ycont_ph = np.angle(self.Ycont, deg=True)

    def converge_modal(self, wv=None, tol=None, Yref=None, Print=False):
        """
        Determine number of modes required to achieve a certain convergence
        of the modal solution in a prescribed frequency range ``wv``. The H-infinity
        norm of the error w.r.t. ``Yref`` is used for assessing convergence.

        .. Warning:: if a reference freq. response, Yref, is not provided, the full-
            state continuous-time frequency response is used as reference. This
            requires the full-states matrices ``Mstr``, ``Cstr``, ``Kstr`` to be available.
        """

        if wv is None:
            wv = self.wv
        assert wv is not None, 'Frequency range not provided.'
        assert tol is not None, 'Tolerance, tol, not provided'
        assert self.modal is True, 'Convergence analysis requires modal=True'

        if Yref is None:
            # use cont. time. full-states as reference
            dlti_here = self.dlti
            self.modal = False
            self.dlti = False
            self.assemble()
            self.freqresp(wv)
            Yref = self.Ycont.copy()
            self.dlti = dlti_here
            self.modal = True

        if Print:
            print('No. modes\tError\tTolerance')
        for nn in range(1, self.Nmodes + 1):
            self.assemble(Nmodes=nn)
            self.freqresp(wv, bode=False)
            Yhere = self.Ycont
            if self.dlti: Yhere = self.Ydisc
            er = np.max(np.abs(Yhere - Yref))
            if Print: print('%.3d\t%.2e\t%.2e' % (nn, er, tol))
            if er < tol:
                if Print: print('Converged!')
                self.Nmodes = nn
                break

    def tune_newmark_damp(self, amplification_factor=0.999):
        """
        Tune artifical damping to achieve a percent reduction of the lower
        frequency (lower damped) mode
        """

        assert self.discr_method == 'newmark' and self.dlti, \
            "select self.discr_method='newmark' and self.dlti=True"

        newmark_damp = self.newmark_damp
        import scipy.optimize as scopt

        def get_res(newmark_damp_log10):
            self.newmark_damp = 10. ** (newmark_damp_log10)
            self.assemble()
            eigsabs = np.abs(np.linalg.eigvals(self.SSdisc.A))
            return np.max(eigsabs) - amplification_factor

        exp_opt = scopt.fsolve(get_res, x0=-3)[0]

        self.newmark_damp = 10. ** exp_opt
        print('artificial viscosity: %.4e' % self.newmark_damp)

    def update_modal(self):
        r"""
        Re-projects the full-states continuous-time structural dynamics equations

        .. math::
            \mathbf{M}\,\mathbf{\ddot{x}} +\mathbf{C}\,\mathbf{\dot{x}} + \mathbf{K\,x} = \mathbf{F}

        onto modal space. The modes used to project are controlled through the
        ``self.proj_modes={damped or undamped}`` attribute.

        .. Warning:: This method overrides SHARPy ``timestep_info`` results and requires
            ``Mstr``, ``Cstr``, ``Kstr`` to be available.

        """

        if self.proj_modes == 'undamped':
            if self.Cstr is not None:
                if self.settings['print_info']:
                    cout.cout_wrap('Warning, projecting system with damping onto undamped modes')

            # Eigenvalues are purely complex - only the complex part is calculated
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(self.Mstr, self.Kstr))

            omega = np.sqrt(eigenvalues)
            order = np.argsort(omega)[:self.Nmodes]
            self.freq_natural = omega[order]

            phi = eigenvectors[:, order]

            # Scale modes to have an identity mass matrix
            dfact = np.diag(np.dot(phi.T, np.dot(self.Mstr, phi)))
            self.U = (1./np.sqrt(dfact))*phi

            # Update
            self.eigs = eigenvalues[order]

            # To do: update SHARPy's timestep info modal results
        else:
            raise NotImplementedError('Projection update for damped systems not yet implemented ')

    def update_truncated_modes(self, nmodes):
        r"""
        Updates the system to the specified number of modes

        Args:
            nmodes:

        Returns:

        """

        # Verify that the new number of modes is less than the current value
        assert nmodes <= self.Nmodes, 'Unable to truncate to %g modes since only %g are available' %(nmodes, self.Nmodes)

        self.Nmodes = nmodes
        self.eigs = self.eigs[:nmodes]
        self.U = self.U[:,:nmodes]
        self.freq_natural = self.freq_natural[:nmodes]
        try:
            self.freq_damp[:nmodes]
        except TypeError:
            pass

        # Update Ccut matrix
        if self.modal:
            self.Ccut = np.dot(self.U.T, np.dot(self.Cstr, self.U))

    def scale_system_normalised_time(self, time_ref):
        r"""
        Scale the system with a normalised time step. The resulting time step is
        :math:`\Delta t = \Delta \bar{t}/t_{ref}`, where the over bar denotes dimensional time.
        The structural equations of motion are rescaled as:

        .. math::
            \mathbf{M}\ddot{\boldsymbol{\eta}} + \mathbf{C} t_{ref} \dot{\boldsymbol{\eta}} + \mathbf{K} t_{ref}^2
            \boldsymbol{\eta} = t_{ref}^2 \mathbf{N}

        For aeroelastic applications, the reference time is usually defined using the semi-chord, :math:`b`, and the
        free stream velocity, :math:`U_\infty`.

        .. math:: t_{ref,ae} = \frac{b}{U_\infty}

        Args:
            time_ref (float): Normalisation factor such that :math:`t/\bar{t}` is non-dimensional.

        """

        if self.scaled_reference_matrices:
            raise UserWarning('System already time scaled. System may just need an update.'
                              ' See update_matrices_time_scale')

        # if time_ref != 1.0 and time_ref is not None:
        if self.num_rig_dof != 0:
            warnings.warn('Time normalisation not yet implemented with rigid body motion.')
        self.scaled_reference_matrices['dt'] = self.dt
        self.dt /= time_ref
        if self.settings['print_info']:
            cout.cout_wrap('Scaling beam according to reduced time...', 0)
            cout.cout_wrap('\tSetting the beam time step to (%.4f)' % self.dt, 1)

        self.scaled_reference_matrices['C'] = self.Cstr.copy()
        self.scaled_reference_matrices['K'] = self.Kstr.copy()
        self.update_matrices_time_scale(time_ref)

    def update_matrices_time_scale(self, time_ref):

        try:
            cout.cout_wrap('Updating C and K matrices and natural frequencies with new normalised time...', 1)
        except ValueError:
            pass

        try:
            self.Kstr = self.scaled_reference_matrices['K'] * time_ref ** 2
            self.Cstr = self.scaled_reference_matrices['C'] * time_ref

            self.freq_natural *= time_ref
        except KeyError:
            raise KeyError('The scaled reference matrices have not been set, most likely because you are trying to '
                           'rescale a dimensional system. Make sure your system is normalised.')

    def cont2disc(self, dt=None):
        """Convert continuous-time SS model into """

        assert self.discr_method is not 'newmark', \
            'For Newmark-beta discretisation, use assemble method directly.'

        if dt is not None:
            self.dt = dt
        else:
            assert self.dt is not None, \
                'Provide time-step for convertion to discrete-time'

        SScont = self.SScont
        tpl = scsig.cont2discrete(
            (SScont.A, SScont.B, SScont.C, SScont.D),
            dt=self.dt, method=self.discr_method)
        self.SSdisc = libss.ss(*tpl[:-1], dt=tpl[-1])
        self.dlti = True


def newmark_ss(Minv, C, K, dt, num_damp=1e-4):
    r"""
    Produces a discrete-time state-space model of the structural equations

    .. math::

        \mathbf{\ddot{x}} &= \mathbf{M}^{-1}( -\mathbf{C}\,\mathbf{\dot{x}}-\mathbf{K}\,\mathbf{x}+\mathbf{F} ) \\
        \mathbf{y} &= \mathbf{x}


    based on the Newmark-:math:`\beta` integration scheme. The output state-space model
    has form:

    .. math::

        \mathbf{X}_{n+1} &= \mathbf{A}\,\mathbf{X}_n + \mathbf{B}\,\mathbf{F}_n \\
        \mathbf{Y} &= \mathbf{C}\,\mathbf{X} + \mathbf{D}\,\mathbf{F}


    with :math:`\mathbf{X} = [\mathbf{x}, \mathbf{\dot{x}}]^T`

    Note that as the state-space representation only requires the input force
    :math:`\mathbf{F}` to be evaluated at time-step :math:`n`,the :math:`\mathbf{C}` and :math:`\mathbf{D}` matrices
    are, in general, fully populated.

    The Newmark-:math:`\beta` integration scheme is carried out following the modifications presented by
    Geradin [1] that render it unconditionally stable. The displacement and velocities are estimated as:

    .. math::
        x_{n+1} &= x_n + \Delta t \dot{x}_n + \left(\frac{1}{2}-\theta_2\right)\Delta t^2 \ddot{x}_n + \theta_2\Delta t
        \ddot{x}_{n+1}  \\
        \dot{x}_{n+1} &= \dot{x}_n + (1-\theta_1)\Delta t \ddot{x}_n + \theta_1\Delta t \ddot{x}_{n+1}

    The stencil is unconditionally stable if the tuning parameters :math:`\theta_1` and :math:`\theta_2` are chosen as:

    .. math::
        \theta_1 &= \frac{1}{2} + \alpha \\
        \theta_2 &= \frac{1}{4} \left(\theta_1 + \frac{1}{2}\right)^2 \\
        \theta_2 &= \frac{5}{80} + \frac{1}{4} (\theta_1 + \theta_1^2) \text{TBC SOURCE}

    where :math:`\alpha>0` accounts for small positive algorithmic damping.

    The following steps describe how to apply the Newmark-beta scheme to a state-space formulation. The original idea
    is based on [1].

    The equation of a second order system dynamics reads:

    .. math::
        M\mathbf{\ddot q} + C\mathbf{\dot q} + K\mathbf{q} = F

    Applying that equation to the time steps :math:`n` and  :math:`n+1`, rearranging terms and multiplying by
    :math:`M^{-1}`:

    .. math::
        \mathbf{\ddot q}_{n} = - M^{-1}C\mathbf{\dot q}_{n} - M^{-1}K\mathbf{q}_{n} + M^{-1}F_{n} \\
        \mathbf{\ddot q}_{n+1} = - M^{-1}C\mathbf{\dot q}_{n+1} - M^{-1}K\mathbf{q}_{n+1} + M^{-1}F_{n+1}

    The relations of the Newmark-beta scheme are:

    .. math::
        \mathbf{q}_{n+1} &= \mathbf{q}_n + \mathbf{\dot q}_n\Delta t +
        (\frac{1}{2}-\beta)\mathbf{\ddot q}_n \Delta t^2 + \beta \mathbf{\ddot q}_{n+1} \Delta t^2 + O(\Delta t^3) \\
        \mathbf{\dot q}_{n+1} &= \mathbf{\dot q}_n + (1-\gamma)\mathbf{\ddot q}_n \Delta t +
        \gamma \mathbf{\ddot q}_{n+1} \Delta t + O(\Delta t^3)

    Substituting the former relation onto the later ones, rearranging terms, and writing it in state-space form:

    .. math::
        \begin{bmatrix} I + M^{-1}K \Delta t^2\beta \quad \Delta t^2\beta M^{-1}C \\ (\gamma \Delta t M^{-1}K)
        \quad (I + \gamma \Delta t M^{-1}C) \end{bmatrix} \begin{Bmatrix} \mathbf{\dot q}_{n+1} \\
        \mathbf{\ddot q}_{n+1} \end{Bmatrix} =
        \begin{bmatrix} (I - \Delta t^2(1/2-\beta)M^{-1}K \quad (\Delta t - \Delta t^2(1/2-\beta)M^{-1}C \\
        (-(1-\gamma)\Delta t M^{-1}K \quad (I - (1-\gamma)\Delta tM^{-1}C \end{bmatrix}
        \begin{Bmatrix}  \mathbf{q}_{n} \\ \mathbf{\dot q}_{n} \end{Bmatrix}	+
        \begin{Bmatrix} (\Delta t^2(1/2-\beta) \\ (1-\gamma)\Delta t \end{Bmatrix} M^{-1}F_n+
        \begin{Bmatrix} (\Delta t^2\beta) \\ (\gamma \Delta t) \end{Bmatrix}M^{-1}F_{n+1}

    To understand SHARPy code, it is convenient to apply the following change of notation:

    .. math::
        \textrm{th1} = \gamma \\
        \textrm{th2} = \beta \\
        \textrm{a0} = \Delta t^2 (1/2 -\beta) \\
        \textrm{b0} = \Delta t (1 -\gamma) \\
        \textrm{a1} = \Delta t^2 \beta \\
        \textrm{b1} = \Delta t \gamma \\

    Finally:

    .. math::
        A_{ss1} \begin{Bmatrix} \mathbf{\dot q}_{n+1} \\ \mathbf{\ddot q}_{n+1} \end{Bmatrix} =
        A_{ss0} \begin{Bmatrix} \mathbf{\dot q}_{n} \\ \mathbf{\ddot q}_{n} \end{Bmatrix} +
        \begin{Bmatrix} (\Delta t^2(1/2-\beta) \\ (1-\gamma)\Delta t \end{Bmatrix} M^{-1}F_n+
        \begin{Bmatrix} (\Delta t^2\beta) \\ (\gamma \Delta t) \end{Bmatrix}M^{-1}F_{n+1}

    To finally isolate the vector at :math:`n+1`, instead of inverting the :math:`A_{ss1}` matrix, several systems are
    solved. Moreover, the output equation is simply :math:`y=x`.

    Args:
        Minv (np.array): Inverse mass matrix :math:`\mathbf{M^{-1}}`
        C (np.array): Damping matrix :math:`\mathbf{C}`
        K (np.array): Stiffness matrix :math:`\mathbf{K}`
        dt (float): Timestep increment
        num_damp (float): Numerical damping. Default ``1e-4``

    Returns:
        tuple: the A, B, C, D matrices of the state space packed in a tuple with the predictor and delay term removed.

    References:
        [1] - Geradin M., Rixen D. - Mechanical Vibrations: Theory and application to structural dynamics
    """

    # weights
    th1 = 0.5 + num_damp
    # th2=0.25*(th1+.5)**2
    th2 = 0.0625 + 0.25 * (th1 + th1 ** 2)

    dt2 = dt ** 2
    a1 = th2 * dt2
    a0 = 0.5 * dt2 - a1
    b1 = th1 * dt
    b0 = dt - b1

    # relevant matrices
    N = K.shape[0]
    Imat = np.eye(N)
    MinvK = np.dot(Minv, K)
    MinvC = np.dot(Minv, C)

    # build ss
    Ass0 = np.block([[Imat - a0 * MinvK, dt * Imat - a0 * MinvC],
                     [-b0 * MinvK, Imat - b0 * MinvC]])
    Ass1 = np.block([[Imat + a1 * MinvK, a1 * MinvC],
                     [b1 * MinvK, Imat + b1 * MinvC]])
    Ass = np.linalg.solve(Ass1, Ass0)

    Bss0 = np.linalg.solve(Ass1, np.block([[a0 * Minv], [b0 * Minv]]))
    Bss1 = np.linalg.solve(Ass1, np.block([[a1 * Minv], [b1 * Minv]]))

    # eliminate predictior term Bss1
    return libss.SSconv(Ass, Bss0, Bss1, C=np.eye(2 * N), D=np.zeros((2 * N, N)))


def sort_eigvals(eigv, eigabsv, tol=1e-6):
    """ sort by magnitude (frequency) and imaginary part if complex conj """

    order = np.argsort(np.abs(eigv))
    eigv = eigv[order]

    for ii in range(len(eigv) - 1):
        # check if ii and ii+1 are the same eigenvalue
        if np.abs(eigv[ii].imag + eigv[ii + 1].imag) / eigabsv[ii] < tol:
            if np.abs(eigv[ii].real - eigv[ii + 1].real) / eigabsv[ii] < tol:

                # swap if required
                if eigv[ii].imag > eigv[ii + 1].imag:
                    temp = eigv[ii]
                    eigv[ii] = eigv[ii + 1]
                    eigv[ii + 1] = temp

                    temp = order[ii]
                    order[ii] = order[ii + 1]
                    order[ii + 1] = temp

    return order, eigv
