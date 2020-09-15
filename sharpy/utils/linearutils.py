"""Linear state-space vector manipulation utilities"""
import numpy as np
import sharpy.utils.algebra as algebra


def structural_vector_to_timestep(vector, tstruct, structure, phi=None, num_rig_dof=0, copy_tstep=True):
    """Transform a state-space structural vector into a time step object

    This adds to a reference time step the following variables extracted from the vector:

        * ``pos`` and ``pos_dot``
        * ``psi`` and ``psi_dot``
        * ``quat``
        * ``for_pos`` and ``for_vel``

    The rest of the structural time step variables are left unchanged

    Args:
        vector(np.ndarray): Vector to plot
        tstruct (sharpy.utils.datastructures.StructTimeStepInfo): Reference time step
        structure (sharpy.structure.models.beam.Beam): Structural information class
        phi (np.ndarray, optional): Eigenvector matrix to transform vector back to nodal coordinates
        num_rig_dof (int): Number of rigid degrees of freedom
        copy_tstep (bool): Return a copy of the reference time step. Else, modify the input one.

    Returns:
        sharpy.utils.datastructures.StructTimeStepInfo: Time step with the aforementioned variables populated from
          state-space vector.
    """
    n_dof = vector.shape[0]
    v = vector[:n_dof//2]
    v_dot = vector[n_dof//2:]
    if phi is not None:  # modal coordinates
        eta = phi.dot(v)
        eta_dot = phi.dot(v_dot)
    else:
        eta = v
        eta_dot = v_dot

    if num_rig_dof != 0:
        eta = eta[:-num_rig_dof]
        eta_dot = eta_dot[:-num_rig_dof]
        beta = eta_dot[-num_rig_dof:]
        beta_bar = np.zeros_like(beta)
    else:
        beta = np.array([])
        beta_bar = np.array([])

    if copy_tstep:
        tstep = tstruct.copy()
    else:
        tstep = tstruct

    vdof = structure.vdof
    num_dof = 6*sum(vdof >= 0)

    q = np.zeros((num_dof + num_rig_dof))
    dqdt = np.zeros_like(q)
    dqddt = np.zeros_like(q)

    pos = np.zeros_like(tstep.pos)
    pos_dot = np.zeros_like(tstep.pos_dot)

    psi = np.zeros_like(tstep.psi)
    psi_dot = np.zeros_like(tstep.psi_dot)

    for_pos = np.zeros_like(tstep.for_pos)
    for_vel = np.zeros_like(tstep.for_vel)
    for_acc = np.zeros_like(tstep.for_acc)
    quat = np.zeros_like(tstep.quat)

    q[:num_dof + num_rig_dof] = np.concatenate((eta, beta_bar))
    dqdt[:num_dof + num_rig_dof] = np.concatenate((eta_dot, beta))

    for i_node in vdof[vdof >= 0]:
        pos[i_node + 1, :] = q[6*i_node: 6*i_node + 3]
        pos_dot[i_node + 1, :] = dqdt[6*i_node + 0: 6*i_node + 3]

    for i_elem in range(tstep.num_elem):
        for i_node in range(tstep.num_node_elem):
            psi[i_elem, i_node, :] = np.linalg.inv(algebra.crv2tan(tstep.psi[i_elem, i_node]).T).dot(q[i_node + 3: i_node + 6])
            psi_dot[i_elem, i_node, :] = dqdt[i_node + 3: i_node + 6]

    if num_rig_dof != 0: # beam is clamped
        for_vel = beta[:6]
        if num_rig_dof == 9:
            quat = algebra.euler2quat(beta[-3:])
        elif num_rig_dof == 10:
            quat = beta[-4:]
        else:
            raise NotImplementedError('Structural vector to timestep for cases without 9 or 10 rigid'
                                      'degrees of freedom not yet supported.')

    tstep.q[:len(q)] += q
    tstep.dqdt[:len(q)] += dqdt

    tstep.pos += pos
    tstep.pos_dot += pos_dot

    tstep.psi += psi
    tstep.psi_dot += psi_dot

    tstep.for_pos += for_pos
    tstep.for_vel += for_vel

    tstep.quat += quat
    tstep.quat /= np.linalg.norm(tstep.quat)  # normalise quaternion

    return tstep

