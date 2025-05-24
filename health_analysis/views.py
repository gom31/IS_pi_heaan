# health_analysis/views.py

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

# 전역 분석기 인스턴스 (싱글톤 패턴)
health_analyzer = None

def get_health_analyzer():
    """싱글톤 패턴으로 분석기 인스턴스 관리"""
    global health_analyzer
    if health_analyzer is None:
        try:
            from .he_utils import get_health_analyzer as get_he_analyzer
            health_analyzer = get_he_analyzer()
            logger.info("Health analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize health analyzer: {e}")
            # 모델이 없어도 기본 응답은 가능하도록
            health_analyzer = None
    return health_analyzer

class HomeView(View):
    """메인 페이지"""
    def get(self, request):
        return render(request, 'health_analysis/index.html')

@method_decorator(csrf_exempt, name='dispatch')
class HealthAnalysisView(View):
    """건강 위험도 분석 API"""
    
    def get(self, request):
        """API 상태 확인"""
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
        """위험도 분석 수행"""
        try:
            # Content-Type 확인
            if request.content_type != 'application/json':
                return JsonResponse({
                    'success': False,
                    'error': 'Content-Type must be application/json',
                    'message': '잘못된 요청 형식입니다.'
                }, status=400)
            
            # JSON 데이터 파싱
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid JSON: {str(e)}',
                    'message': 'JSON 파싱 오류가 발생했습니다.'
                }, status=400)
            
            # 필수 필드 검증
            required_fields = [
                'sex', 'age', 'edu_lvl', 'had_sex', 'n_s_part', 
                'con_use', 'r_use_con', 'r_sea', 'r_have_1sp', 
                'r_nhave_sex', 'hiv_mosq', 'h_sti', 'h_o_sti',
                'e_t_hiv', 'p_t_hiv', 's_test', 't_in_lab', 'f_t_resu'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required fields: {missing_fields}',
                    'message': f'필수 필드가 누락되었습니다: {missing_fields}'
                }, status=400)
            
            # 입력 데이터 검증 및 정리
            user_inputs = {}
            validation_errors = []
            
            for field in required_fields:
                value = data.get(field)
                
                # 숫자 검증
                try:
                    if isinstance(value, str):
                        value = float(value)
                    elif value is None:
                        value = 0
                    
                    user_inputs[field] = int(value)
                    
                    # 범위 검증 (대부분 0 또는 1의 이진값)
                    if field in ['sex', 'had_sex', 'con_use', 'r_use_con', 'r_sea', 
                                'r_have_1sp', 'r_nhave_sex', 'hiv_mosq', 'h_sti', 
                                'h_o_sti', 'e_t_hiv', 'p_t_hiv', 's_test', 't_in_lab', 'f_t_resu']:
                        if user_inputs[field] not in [0, 1]:
                            validation_errors.append(f'{field}는 0 또는 1이어야 합니다.')
                    
                    elif field == 'age':
                        if user_inputs[field] not in range(1, 8):  # 1-7
                            validation_errors.append('age는 1-7 범위여야 합니다.')
                    
                    elif field == 'edu_lvl':
                        if user_inputs[field] not in range(0, 5):  # 0-4
                            validation_errors.append('edu_lvl은 0-4 범위여야 합니다.')
                    
                    elif field == 'n_s_part':
                        if user_inputs[field] < 0 or user_inputs[field] > 100:
                            validation_errors.append('n_s_part는 0-100 범위여야 합니다.')
                    
                except (ValueError, TypeError):
                    validation_errors.append(f'{field}는 유효한 숫자여야 합니다.')
            
            if validation_errors:
                return JsonResponse({
                    'success': False,
                    'error': 'Validation errors',
                    'validation_errors': validation_errors,
                    'message': '입력 데이터 검증 오류가 발생했습니다.'
                }, status=400)
            
            # 동형 암호 분석 수행
            try:
                analyzer = get_health_analyzer()
                
                if analyzer is None:
                    return JsonResponse({
                        'success': False,
                        'error': 'Model not available',
                        'message': '모델이 로드되지 않았습니다. 먼저 모델을 훈련해주세요.'
                    }, status=503)
                
                result = analyzer.analyze_health_risk(user_inputs, use_he=True)
                
                logger.info(f"Analysis completed for user input: {user_inputs}")
                
                return JsonResponse({
                    'success': True,
                    'data': result,
                    'message': '분석이 성공적으로 완료되었습니다.',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as analysis_error:
                logger.error(f"Analysis error: {analysis_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                return JsonResponse({
                    'success': False,
                    'error': f'Analysis failed: {str(analysis_error)}',
                    'message': '분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
                }, status=500)
            
        except Exception as e:
            logger.error(f"Unexpected error in HealthAnalysisView: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return JsonResponse({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'message': '서버 내부 오류가 발생했습니다.'
            }, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class HealthFormView(View):
    """건강 설문 양식 제공"""
    
    def get(self, request):
        """설문 양식 구조 반환"""
        form_structure = {
            'title': 'SSH - 익명 성적 건강 위험도 평가',
            'description': '개인정보 보호를 위해 동형암호를 사용하는 익명 건강 평가 서비스입니다.',
            'privacy_notice': '모든 데이터는 암호화되어 처리되며, 개인을 식별할 수 있는 정보는 저장되지 않습니다.',
            'fields': [
                {
                    'name': 'sex',
                    'label': '성별',
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': '여성'},
                        {'value': 1, 'label': '남성'}
                    ],
                    'required': True,
                    'description': '생물학적 성별을 선택해주세요.'
                },
                {
                    'name': 'age',
                    'label': '연령대',
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': '15-19세'},
                        {'value': 2, 'label': '20-24세'},
                        {'value': 3, 'label': '25-29세'},
                        {'value': 4, 'label': '30-34세'},
                        {'value': 5, 'label': '35-39세'},
                        {'value': 6, 'label': '40-44세'},
                        {'value': 7, 'label': '45-49세'}
                    ],
                    'required': True,
                    'description': '현재 연령대를 선택해주세요.'
                },
                # ... 나머지 필드들은 이전과 동일
            ]
        }
        
        return JsonResponse({
            'success': True,
            'data': form_structure
        })

class PrivacyPolicyView(View):
    """개인정보처리방침"""
    def get(self, request):
        return render(request, 'health_analysis/privacy.html')

class AboutView(View):
    """서비스 소개"""
    def get(self, request):
        return render(request, 'health_analysis/about.html')

@method_decorator(csrf_exempt, name='dispatch')
class HealthTestView(View):
    """개발/테스트용 API"""
    
    def get(self, request):
        """테스트 페이지"""
        return JsonResponse({
            'success': True,
            'message': 'Test API is working',
            'test_data_available': True
        })
    
    def post(self, request):
        """테스트 데이터로 분석 수행"""
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
            'f_t_resu': 0
        }
        
        try:
            analyzer = get_health_analyzer()
            
            if analyzer is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Model not available',
                    'message': '모델이 로드되지 않았습니다.',
                    'test_input': test_data
                }, status=503)
            
            result = analyzer.analyze_health_risk(test_data, use_he=True)
            
            return JsonResponse({
                'success': True,
                'data': result,
                'test_input': test_data,
                'message': '테스트 분석이 완료되었습니다.'
            })
            
        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'test_input': test_data,
                'message': '테스트 중 오류가 발생했습니다.'
            }, status=500)