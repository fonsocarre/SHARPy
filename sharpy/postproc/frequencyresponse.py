import numpy as np
import time
import os
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout
import warnings
import sharpy.linear.src.libss as libss
import h5py as h5
import sharpy.utils.frequencyutils as frequencyutils
from sharpy.utils.frequencyutils import find_target_system


@solver_interface.solver
class FrequencyResponse(solver_interface.BaseSolver):
    """
    Frequency Response Calculator.

    Computes the frequency response of a built linear system. The frequency will be calculated for the systems
    specified in the ``target_system`` list. The desired ``frequency_unit`` will be either ``w`` for radians/s or ``k``
    for reduced frequency (if the system is scaled). The ``frequency_bounds`` setting will set the lower and upper
    bounds of the response, while ``num_freqs`` will specify the number of evaluations.
    The option ``frequency_spacing`` allows you to space the evaluations point following a ``log``
    or ``linear`` spacing.

    If ``compute_hinf`` is set, the H-infinity norm of the system is calculated.

    This will be saved to a binary ``.h5`` file as detailed in :func:`save_freq_resp`.

    Finally, the ``quick_plot`` option will plot some quick and dirty bode plots of the response. This requires
    access to ``matplotlib``.

    """
    solver_id = 'FrequencyResponse'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder.'

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Write output to screen.'

    settings_types['target_system'] = 'list(str)'
    settings_default['target_system'] = ['aeroelastic']
    settings_description['target_system'] = 'System or systems for which to find frequency response.'
    settings_options['target_system'] = ['aeroelastic', 'aerodynamic', 'structural']

    settings_types['frequency_unit'] = 'str'
    settings_default['frequency_unit'] = 'k'
    settings_description['frequency_unit'] = 'Units of frequency, ``w`` for rad/s, ``k`` reduced.'
    settings_options['frequency_unit'] = ['w', 'k']

    settings_types['frequency_scaling'] = 'dict'
    settings_default['frequency_scaling'] = {}
    settings_description['frequency_scaling'] = 'Dictionary containing the frequency scaling factors, if the ' \
                                                'aerodynamic system has not been previously scaled. Applied also if ' \
                                                'the desired unit is reduced frequency.'

    scaling_types = dict()
    scaling_default = dict()
    scaling_description = dict()

    scaling_types['length'] = 'float'
    scaling_default['length'] = 1.
    scaling_description['length'] = 'Length scaling factor.'

    scaling_types['speed'] = 'float'
    scaling_default['speed'] = 1.
    scaling_description['speed'] = 'Speed scaling factor.'

    settings_types['frequency_bounds'] = 'list(float)'
    settings_default['frequency_bounds'] = [1e-3, 1]
    settings_description['frequency_bounds'] = 'Lower and upper frequency bounds in the corresponding unit.'

    settings_types['frequency_spacing'] = 'str'
    settings_default['frequency_spacing'] = 'linear'
    settings_description['frequency_spacing'] = 'Compute the frequency response in a ``linear`` or ``log`` grid.'
    settings_options['frequency_spacing'] = ['linear', 'log']

    settings_types['num_freqs'] = 'int'
    settings_default['num_freqs'] = 50
    settings_description['num_freqs'] = 'Number of frequencies to evaluate.'

    settings_types['compute_hinf'] = 'bool'
    settings_default['compute_hinf'] = False
    settings_description['compute_hinf'] = 'Compute Hinfinity norm of the system.'

    settings_types['quick_plot'] = 'bool'
    settings_default['quick_plot'] = False
    settings_description['quick_plot'] = 'Produce array of ``.png`` plots showing response. Requires matplotlib.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    scaling_table = settings_utils.SettingsTable()
    __doc__ += scaling_table.generate(scaling_types, scaling_default, scaling_description,
                                      header_line='The scaling dictionary takes the following entries:')

    def __init__(self):

        self.settings = None
        self.data = None
        self.folder = None
        self.print_info = False

        self.scaled = False
        self.w_to_k = 1
        self.wv = None

    def initialise(self, data, custom_settings=None):

        self.data = data

        if not custom_settings:
            self.settings = self.data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                       self.settings_options,
                                       no_ctype=True)

        self.print_info = self.settings['print_info']

        if self.settings['frequency_unit'] == 'k':
            self.scaled = True
            if self.data.linear.linear_system.uvlm.scaled:
                scaling = self.data.linear.linear_system.uvlm.sys.ScalingFacts
            else:
                scaling = self.settings['frequency_scaling']
                settings_utils.to_custom_types(scaling, self.scaling_types, self.scaling_default,
                                               no_ctype=True)
            self.w_to_k = scaling['length'] / scaling['speed']
        else:
            self.w_to_k = 1.

        # Frequency bounds in radians
        lb = self.settings['frequency_bounds'][0] / self.w_to_k
        ub = self.settings['frequency_bounds'][1] / self.w_to_k

        nfreqs = self.settings['num_freqs']
        if self.settings['frequency_spacing'] == 'linear':
            self.wv = np.linspace(lb, ub, nfreqs)
        elif self.settings['frequency_spacing'] == 'log':
            self.wv = np.logspace(np.log10(lb), np.log10(ub), nfreqs)
        else:
            raise NotImplementedError('Unrecognised frequency spacing setting %s' % self.settings['frequency_spacing'])

        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/frequencyresponse/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, ss=None):
        """
        Computes the frequency response of the linear state-space.

        Args:
            ss (sharpy.linear.src.libss.ss (Optional)): State-space object for which to compute the frequency response.
              If not given, the response for the previously assembled systems and specified in ``target_system`` will
              be performed.

        """

        if ss is None:
            ss_list = [find_target_system(self.data, system_name) for system_name in self.settings['target_system']]
        elif type(ss) is libss.ss:
            ss_list = [ss]
        elif type(ss) is list:
            ss_list = ss
        else:
            raise TypeError('ss input must be either a libss.ss instance or a list of libss.ss')

        for ith, system in enumerate(ss_list):
            if self.print_info:
                cout.cout_wrap('Computing frequency response...')
            if ss is None:
                try:
                    system_name = self.settings['target_system'][ith]
                    if self.print_info:
                        cout.cout_wrap('\tComputing frequency response for %s system' % system_name, 1)
                except IndexError:
                    system_name = None
            else:
                system_name = None  # For the case where the state-space is parsed in run().

            t0fom = time.time()
            y_freq_fom = system.freqresp(self.wv)
            tfom = time.time() - t0fom

            if self.settings['compute_hinf']:

                if self.print_info:
                    cout.cout_wrap('Computing H-infinity norm...')
                try:
                    hinf = frequencyutils.h_infinity_norm(system, iter_max=50, print_info=self.settings['print_info'])
                except np.linalg.LinAlgError:
                    hinf = None
                    cout.cout_wrap('H-infinity calculation did not converge', 4)

            else:
                hinf = None

            self.save_freq_resp(self.wv * self.w_to_k, y_freq_fom, system_name=system_name, hinf=hinf)

            cout.cout_wrap('\tComputed the frequency response in %f s' % tfom, 2)

            if self.settings['quick_plot']:
                self.quick_plot(y_freq_fom, subfolder=system_name)

        return self.data

    def save_freq_resp(self, wv, Yfreq, system_name=None, hinf=None):
        """
        Saves the frequency response to a binary ``.h5`` file.

        If the system has not been scaled, the units of frequency are ``rad/s`` and the response is given in complex
        form. The response is saved in a ``[p, m, n_freq_eval]`` format, where ``p`` corresponds to the system's
        outputs, ``n`` to the number of inputs and ``n_freq_eval`` to the number of frequency evaluations.

        Args:
            wv (np.ndarray): Frequency array.
            Y_freq (np.ndarray): Frequency response data ``[p, m, n_freq_eval]`` matrix.
            system_name (str (optional)): State-space system name.
            hinf (float (optional)): H-infinity norm of the system.
        """

        with open(self.folder + '/freqdata_readme.txt', 'w') as outfile:
            outfile.write('Frequency Response Data Output\n\n')
            outfile.write('Frequency data found in the relevant .h5 file\n')
            outfile.write('The units of frequency are rad/s\nThe frequency' \
                          'response is given in complex form.')

        case_name = ''
        if system_name is not None:
            case_name += system_name + '.'

        p, m, _ = Yfreq.shape

        h5filename = self.folder + '/' + case_name + 'freqresp.h5'
        with h5.File(h5filename, 'w') as f:
            f.create_dataset('frequency', data=wv)
            f.create_dataset('response', data=Yfreq, dtype=complex)
            f.create_dataset('inputs', data=m)
            f.create_dataset('outputs', data=p)
            if hinf is not None:
                f.create_dataset('hinf_norm', data=hinf)

        if self.print_info:
            cout.cout_wrap('Saved .h5 file to %s with frequency response data' % h5filename)

    def quick_plot(self, y_freq_fom=None, subfolder=None):
        p, m, _ = y_freq_fom.shape
        try:
            cout.cout_wrap('\tCreating Quick plots of the frequency response', 1)

            out_folder = self.folder
            if subfolder:
                out_folder += '/' + subfolder

            if not os.path.isdir(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            import matplotlib.pyplot as plt
            for mj in range(m):
                for pj in range(p):
                    fig1, ax1 = plt.subplots(nrows=2)
                    fig_title = 'in%02g_out%02g' % (mj, pj)
                    ax1[0].set_title(fig_title)
                    if y_freq_fom is not None:
                        ax1[0].plot(self.wv * self.w_to_k, 20 * np.log10(np.abs(y_freq_fom[pj, mj, :])), color='C0')
                        ax1[1].plot(self.wv * self.w_to_k, np.angle(y_freq_fom[pj, mj, :]), '-', color='C0')
                    if self.scaled:
                        ax1[1].set_xlabel('Reduced Frequency, k [-]')
                    else:
                        ax1[1].set_xlabel(r'Frequency, $\omega$ [rad/s]')

                    ax1[0].set_ylabel('Amplitude [dB]')
                    ax1[1].set_ylabel('Phase [rad]')
                    fig1.savefig(out_folder + '/' + fig_title + '.png')
                    plt.close()

            cout.cout_wrap('\tPlots saved to %s' % out_folder, 1)
        except ModuleNotFoundError:
            warnings.warn('Matplotlib not found - skipping plot')
