from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
import logging

logger = logging.getLogger(__name__)

class KMCSimulationForm(forms.Form):
    lattice_size = forms.IntegerField(
        label="Lattice Size",
        min_value=10,
        max_value=100,
        initial=50,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Size (20-100)',
        })
    )
    steps = forms.IntegerField(
        label="Simulation Steps",
        min_value=100,
        max_value=5000,
        initial=1000,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Steps (100-5000)',
        })
    )
    temperature = forms.TypedChoiceField(
        label="Temperature (°K)",
        choices=[(220, '220 K'), (240, '240 K'), (260, '260 K'), (280, '280 K'), (300, '300 K'), (320, '320 K')],
        initial=260,
        coerce=float,
        widget=forms.Select(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
        })
    )
    impurity_concentration = forms.FloatField(
        label="Impurity Concentration",
        min_value=0.0,
        max_value=0.01,
        initial=0.001,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.0001',
            'placeholder': 'Impurity (0.0-0.01)',
        })
    )
    diffusion_x = forms.FloatField(
        label="Diffusion Barrier X (eV)",
        min_value=0.1,
        max_value=2.0,
        initial=0.75,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.01',
            'placeholder': 'Diffusion X (0.1-2.0 eV)',
        })
    )
    diffusion_y = forms.FloatField(
        label="Diffusion Barrier Y (eV)",
        min_value=0.1,
        max_value=2.0,
        initial=0.95,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.01',
            'placeholder': 'Diffusion Y (0.1-2.0 eV)',
        })
    )
    diffusion_z = forms.FloatField(
        label="Diffusion Barrier Z (eV)",
        min_value=0.1,
        max_value=2.0,
        initial=1.2,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.01',
            'placeholder': 'Diffusion Z (0.1-2.0 eV)',
        })
    )
    impurity_effect = forms.FloatField(
        label="Impurity Effect on Diffusion (eV)",
        min_value=0.0,
        max_value=1.0,
        initial=0.2,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.01',
            'placeholder': 'Impurity Effect (0-1.0 eV)',
        })
    )
    flux = forms.TypedChoiceField(
        label="Flux (ML/s)",
        choices=[(0.004, '0.004 ML/s'), (0.006, '0.006 ML/s'), (0.02, '0.02 ML/s'), (0.04, '0.04 ML/s'), (0.08, '0.08 ML/s'), (0.3, '0.3 ML/s')],
        initial=0.004,
        coerce=float,
        widget=forms.Select(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
        })
    )
    name = forms.CharField(
        label="Simulation Name",
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Enter simulation name (optional)',
            'id': 'id_name'
        }),
        help_text="Optional name for the simulation"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Run Simulation', css_class='w-full py-3 px-6 btn-primary text-white font-semibold rounded-lg'))

    def clean(self):
        cleaned_data = super().clean()
        lattice_size = cleaned_data.get('lattice_size')
        impurity_concentration = cleaned_data.get('impurity_concentration')
        name = cleaned_data.get('name')
        if lattice_size and impurity_concentration is not None:
            total_sites = lattice_size ** 3
            actual_impurities = int(total_sites * impurity_concentration)
            if actual_impurities < 10:
                self._errors['impurity_concentration'] = self.error_class([
                    f"Warning: Impurity concentration results in {actual_impurities} impurities. Consider increasing to achieve ~10 impurities."
                ])
        if name and len(name.strip()) == 0:
            self._errors['name'] = self.error_class([
                "Simulation name cannot be empty or contain only whitespace."
            ])
        return cleaned_data