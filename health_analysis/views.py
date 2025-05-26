from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
import json
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

# Global analyzer instance (singleton pattern)
health_analyzer = None

def get_health_analyzer():
    """Manage analyzer instance using singleton pattern"""
    global health_analyzer
    if health_analyzer is None:
        try:
            from .he_utils import get_health_analyzer as get_he_analyzer
            health_analyzer = get_he_analyzer()
            logger.info("Health analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize health analyzer: {e}")
            # Allow basic response even if the model is missing
            health_analyzer = None
    return health_analyzer

class HomeView(View):
    """Main page"""
    def get(self, request):
        return render(request, 'health_analysis/index.html')

@method_decorator(csrf_exempt, name='dispatch')
class HealthAnalysisView(View):
    """Health risk analysis API"""

    def get(self, request):
        """API status check"""
        try:
            analyzer = get_health_analyzer()

            if analyzer is None:
                return JsonResponse({
                    'success': False,
                    'message': 'Health Analysis API is not ready',
                    'service': 'SSH (SecureSexHealth)',
                    'error': 'Model not loaded',
                    'status': 'unhealthy'
                }, status=503)

            return JsonResponse({
                'success': True,
                'message': 'Health Analysis API is running',
                'service': 'SSH (SecureSexHealth)',
                'model_info': {
                    'type': getattr(analyzer, 'model_loader', {}).get('metadata', {}).get('model_info', {}).get('type', 'Unknown'),
                    'features': len(getattr(analyzer, 'feature_names', [])),
                    'status': 'loaded'
                },
                'status': 'healthy'
            })

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JsonResponse({
                'success': False,
                'message': 'Health Analysis API encountered an error',
                'error': str(e),
                'status': 'error'
            }, status=500)

    def post(self, request):
        """Perform risk analysis"""
        try:
            # Check Content-Type
            if request.content_type != 'application/json':
                return JsonResponse({
                    'success': False,
                    'error': 'Content-Type must be application/json',
                    'message': 'Invalid request format.'
                }, status=400)

            # Parse JSON data
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid JSON: {str(e)}',
                    'message': 'JSON parsing error occurred.'
                }, status=400)

            # Required field validation
            required_fields = [
                'sex', 'age', 'edu_lvl', 'had_sex', 'n_s_part', 
                'con_use', 'r_use_con', 'r_sea', 'r_have_1sp', 
                'r_nhave_sex', 'hiv_mosq', 'h_sti', 'h_o_sti',
                'e_t_hiv', 'p_t_hiv', 's_test', 't_in_lab', 'h_aids'
            ]

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required fields: {missing_fields}',
                    'message': f'Required fields are missing: {missing_fields}'
                }, status=400)

            # Validate and clean input data
            user_inputs = {}
            validation_errors = []

            for field in required_fields:
                value = data.get(field)

                # Number validation
                try:
                    if isinstance(value, str):
                        value = float(value)
                    elif value is None:
                        value = 0

                    user_inputs[field] = int(value)

                    # Range validation (most fields are binary 0 or 1)
                    if field in ['sex', 'had_sex', 'con_use', 'r_use_con', 'r_sea', 
                                'r_have_1sp', 'r_nhave_sex', 'hiv_mosq', 'h_sti', 
                                'h_o_sti', 'e_t_hiv', 'p_t_hiv', 's_test', 't_in_lab', 'h_aids']:
                        if user_inputs[field] not in [0, 1]:
                            validation_errors.append(f'{field} must be 0 or 1.')

                    elif field == 'age':
                        if user_inputs[field] not in range(1, 8):  # 1-7
                            validation_errors.append('age must be in range 1-7.')

                    elif field == 'edu_lvl':
                        if user_inputs[field] not in range(0, 5):  # 0-4
                            validation_errors.append('edu_lvl must be in range 0-4.')

                    elif field == 'n_s_part':
                        if user_inputs[field] < 0 or user_inputs[field] > 100:
                            validation_errors.append('n_s_part must be in range 0-100.')

                except (ValueError, TypeError):
                    validation_errors.append(f'{field} must be a valid number.')

            if validation_errors:
                return JsonResponse({
                    'success': False,
                    'error': 'Validation errors',
                    'validation_errors': validation_errors,
                    'message': 'Input data validation failed.'
                }, status=400)

            # Perform homomorphic encryption analysis
            try:
                analyzer = get_health_analyzer()

                if analyzer is None:
                    return JsonResponse({
                        'success': False,
                        'error': 'Model not available',
                        'message': 'Model is not loaded. Please train the model first.'
                    }, status=503)

                result = analyzer.analyze_health_risk(user_inputs, use_he=True)

                logger.info(f"Analysis completed for user input: {user_inputs}")

                return JsonResponse({
                    'success': True,
                    'data': result,
                    'message': 'Analysis completed successfully.',
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as analysis_error:
                logger.error(f"Analysis error: {analysis_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                return JsonResponse({
                    'success': False,
                    'error': f'Analysis failed: {str(analysis_error)}',
                    'message': 'An error occurred during analysis. Please try again later.'
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in HealthAnalysisView: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            return JsonResponse({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'message': 'An internal server error occurred.'
            }, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class HealthFormView(View):
    """Provide health survey form"""

    def get(self, request):
        """Return survey form structure"""
        form_structure = {
            'title': 'SSH - Anonymous Sexual Health Risk Assessment',
            'description': 'An anonymous health evaluation service using homomorphic encryption to protect personal information.',
            'privacy_notice': 'All data is processed in encrypted form and no personally identifiable information is stored.',
            'fields': [
                {
                    'name': 'sex',
                    'label': 'Gender',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'Female'},
                        {'value': 1, 'label': 'Male'}
                    ],
                    'required': True,
                    'description': 'Please select your biological sex.'
                },
                {
                    'name': 'age',
                    'label': 'Age Group',
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': '15-19 years'},
                        {'value': 2, 'label': '20-24 years'},
                        {'value': 3, 'label': '25-29 years'},
                        {'value': 4, 'label': '30-34 years'},
                        {'value': 5, 'label': '35-39 years'},
                        {'value': 6, 'label': '40-44 years'},
                        {'value': 7, 'label': '45-49 years'}
                    ],
                    'required': True,
                    'description': 'Please select your current age group.'
                },
                {
                    'name': 'edu_lvl',
                    'label': 'Education Level',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'Elementary school or below'},
                        {'value': 1, 'label': 'Middle school graduate'},
                        {'value': 2, 'label': 'High school graduate'},
                        {'value': 3, 'label': 'College graduate'},
                        {'value': 4, 'label': 'Graduate degree or higher'}
                    ],
                    'required': True,
                    'description': 'Please select your highest education level.'
                },
                {
                    'name': 'had_sex',
                    'label': 'Sexual Experience',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you ever had sexual intercourse?'
                },
                {
                    'name': 'n_s_part',
                    'label': 'Number of Sexual Partners',
                    'type': 'number',
                    'min': 0,
                    'max': 100,
                    'required': True,
                    'description': 'Please enter the total number of sexual partners you have had (approximate number).'
                },
                {
                    'name': 'con_use',
                    'label': 'Condom Use',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Do you use condoms during sexual intercourse?'
                },
                {
                    'name': 'r_use_con',
                    'label': 'Recent Condom Use',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Did you use a condom during your most recent sexual encounter?'
                },
                {
                    'name': 'r_sea',
                    'label': 'Recent Sexual Activity',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you been sexually active in the past 12 months?'
                },
                {
                    'name': 'r_have_1sp',
                    'label': 'Single Partner Recently',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'In the past 12 months, have you had only one sexual partner?'
                },
                {
                    'name': 'r_nhave_sex',
                    'label': 'No Recent Sexual Activity',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you had no sexual activity in the past 12 months?'
                },
                {
                    'name': 'hiv_mosq',
                    'label': 'HIV Transmission Knowledge (Mosquito)',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No / Don\'t know'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Do you think HIV can be transmitted through mosquito bites?'
                },
                {
                    'name': 'h_sti',
                    'label': 'Heard of STI',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you heard of sexually transmitted infections (STIs)?'
                },
                {
                    'name': 'h_o_sti',
                    'label': 'Heard of Other STIs',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you heard of STIs other than HIV/AIDS?'
                },
                {
                    'name': 'e_t_hiv',
                    'label': 'Ever Tested for HIV',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you ever been tested for HIV?'
                },
                {
                    'name': 'p_t_hiv',
                    'label': 'Partner Tested for HIV',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No / Don\'t know'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Has your sexual partner ever been tested for HIV?'
                },
                {
                    'name': 's_test',
                    'label': 'STI Screening',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you ever been screened for sexually transmitted infections?'
                },
                {
                    'name': 't_in_lab',
                    'label': 'Laboratory Testing',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you ever had laboratory tests for sexual health?'
                },
                {
                    'name': 'h_aids',
                    'label': 'Heard of AIDS',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'No'},
                        {'value': 1, 'label': 'Yes'}
                    ],
                    'required': True,
                    'description': 'Have you heard of AIDS?'
                }
            ]
        }

        return JsonResponse({
            'success': True,
            'data': form_structure
        })

class PrivacyPolicyView(View):
    """Privacy policy page"""
    def get(self, request):
        return render(request, 'health_analysis/privacy.html')

class AboutView(View):
    """Service introduction page"""
    def get(self, request):
        return render(request, 'health_analysis/about.html')

@method_decorator(csrf_exempt, name='dispatch')
class HealthTestView(View):
    """Development/Test API"""

    def get(self, request):
        """Test page"""
        return JsonResponse({
            'success': True,
            'message': 'Test API is working',
            'test_data_available': True,
            'endpoint_info': {
                'description': 'Use POST method to test health risk analysis',
                'test_data': 'Predefined test data will be used for analysis'
            }
        })

    def post(self, request):
        """Perform analysis with test data"""
        test_data = {
            'sex': 1,           # Male
            'age': 3,           # 25-29 years
            'edu_lvl': 2,       # High school graduate
            'had_sex': 1,       # Yes
            'n_s_part': 2,      # 2 sexual partners
            'con_use': 1,       # Uses condoms
            'r_use_con': 0,     # Did not use condom recently
            'r_sea': 1,         # Sexually active recently
            'r_have_1sp': 0,    # Multiple partners recently
            'r_nhave_sex': 0,   # Had sexual activity recently
            'hiv_mosq': 0,      # Knows HIV is not transmitted by mosquitos
            'h_sti': 1,         # Heard of STIs
            'h_o_sti': 1,       # Heard of other STIs
            'e_t_hiv': 0,       # Never tested for HIV
            'p_t_hiv': 1,       # Partner tested for HIV
            's_test': 0,        # Never screened for STIs
            't_in_lab': 0,      # No lab tests
            'h_aids': 1         # Heard of AIDS
        }

        try:
            analyzer = get_health_analyzer()

            if analyzer is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Model not available',
                    'message': 'Health analysis model is not loaded. Please ensure the model is properly trained and available.',
                    'test_input': test_data,
                    'recommendations': [
                        'Check if the model training has been completed',
                        'Verify that the HE (Homomorphic Encryption) utils are properly configured',
                        'Ensure all required dependencies are installed'
                    ]
                }, status=503)

            result = analyzer.analyze_health_risk(test_data, use_he=True)

            return JsonResponse({
                'success': True,
                'data': result,
                'test_input': test_data,
                'message': 'Test analysis completed successfully.',
                'note': 'This is a test analysis using predefined data for development purposes.'
            })

        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            logger.error(f"Test traceback: {traceback.format_exc()}")
            
            return JsonResponse({
                'success': False,
                'error': str(e),
                'test_input': test_data,
                'message': 'An error occurred during test analysis.',
                'debug_info': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }, status=500)