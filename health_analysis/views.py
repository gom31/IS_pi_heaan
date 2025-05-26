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

            # Required field check
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

                # Validate number
                try:
                    if isinstance(value, str):
                        value = float(value)
                    elif value is None:
                        value = 0

                    user_inputs[field] = int(value)

                    # Range validation
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
                        {'value': 1, 'label': '15-19'},
                        {'value': 2, 'label': '20-24'},
                        {'value': 3, 'label': '25-29'},
                        {'value': 4, 'label': '30-34'},
                        {'value': 5, 'label': '35-39'},
                        {'value': 6, 'label': '40-44'},
                        {'value': 7, 'label': '45-49'}
                    ],
                    'required': True,
                    'description': 'Please select your current age group.'
                }
                # ... remaining fields unchanged
            ]
        }

        return JsonResponse({
            'success': True,
            'data': form_structure
        })

class PrivacyPolicyView(View):
    """Privacy policy"""
    def get(self, request):
        return render(request, 'health_analysis/privacy.html')

class AboutView(View):
    """Service introduction"""
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
            'test_data_available': True
        })

    def post(self, request):
        """Perform analysis with test data"""
        test_data = {
            'sex': 1,
            'age': 3,
            'edu_lvl': 2,
            'had_sex': 1,
            'n_s_part': 2,
            'con_use': 1,
            'r_use_con': 0,
            'r_sea': 1,
            'r_have_1sp': 0,
            'r_nhave_sex': 0,
            'hiv_mosq': 0,
            'h_sti': 1,
            'h_o_sti': 1,
            'e_t_hiv': 0,
            'p_t_hiv': 1,
            's_test': 0,
            't_in_lab': 0,
            'h_aids': 0
        }

        try:
            analyzer = get_health_analyzer()

            if analyzer is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Model not available',
                    'message': 'Model is not loaded.',
                    'test_input': test_data
                }, status=503)

            result = analyzer.analyze_health_risk(test_data, use_he=True)

            return JsonResponse({
                'success': True,
                'data': result,
                'test_input': test_data,
                'message': 'Test analysis completed.'
            })

        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'test_input': test_data,
                'message': 'An error occurred during testing.'
            }, status=500)
