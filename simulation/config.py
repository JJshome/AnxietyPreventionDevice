"""
불안장애 예방 시스템 시뮬레이션 환경 설정

시뮬레이션 환경에서 사용되는 설정값과 시나리오를 정의합니다.
"""

import os
import json
from enum import Enum
from typing import Dict, List, Any, Optional

# 기본 설정 파일 경로
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "simulation_config.json")


class SimulationScenario(Enum):
    """시뮬레이션 시나리오 열거형"""
    NORMAL = "normal"                 # 정상 심전도 패턴
    MILD_ANXIETY = "mild_anxiety"     # 경미한 불안 패턴
    MODERATE_ANXIETY = "moderate_anxiety"  # 중등도 불안 패턴
    SEVERE_ANXIETY = "severe_anxiety"  # 심각한 불안 패턴
    PANIC = "panic"                   # 패닉 발작 패턴
    RECOVERY = "recovery"             # 자극 후 회복 패턴
    CUSTOM = "custom"                 # 사용자 정의 패턴


class SimulationConfig:
    """시뮬레이션 환경 설정 클래스"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        시뮬레이션 설정 초기화
        
        Args:
            config_file: 설정 파일 경로 (None인 경우 기본 설정 사용)
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """
        설정 파일 로드
        
        Returns:
            설정 딕셔너리
        """
        default_config = {
            "simulation": {
                "duration": 1800,            # 시뮬레이션 기본 지속 시간 (초)
                "real_time_factor": 1.0,     # 실시간 대비 속도 배율 (1.0 = 실시간)
                "headless": False,           # 헤드리스 모드 활성화 여부
                "logging_level": "INFO",     # 로깅 레벨
                "save_results": True,        # 결과 저장 여부
                "results_dir": "./results"   # 결과 저장 디렉토리
            },
            "ecg_simulator": {
                "sample_rate": 250,          # 샘플링 레이트 (Hz)
                "noise_level": 0.03,         # 노이즈 레벨 (0.0 ~ 1.0)
                "baseline_hr": 70,           # 기본 심박수 (BPM)
                "hrv_level": 0.05,           # HRV 변동성 레벨 (0.0 ~ 1.0)
                "artifact_probability": 0.01, # 인공 잡음 발생 확률
                "signal_quality": 0.9,       # 신호 품질 (0.0 ~ 1.0)
                "battery_drain_rate": 0.001  # 배터리 소모율 (%/초)
            },
            "stimulator_simulator": {
                "response_time": 0.2,        # 명령에 대한 응답 지연 시간 (초)
                "connection_reliability": 0.98, # 연결 신뢰도 (0.0 ~ 1.0)
                "battery_drain_rate": 0.002, # 배터리 소모율 (%/초)
                "max_intensity": 4.0,        # 최대 자극 강도 (mA)
                "supported_frequencies": [0.5, 1, 2, 4, 8, 16, 20, 30, 40, 50, 60, 80, 100], # 지원 주파수 (Hz)
                "min_phase_delay": 0.1,      # 최소 위상 지연 (초)
                "max_phase_delay": 1.0       # 최대 위상 지연 (초)
            },
            "scenarios": {
                "normal": {
                    "description": "정상 심전도 패턴",
                    "duration": 600,         # 시나리오 지속 시간 (초)
                    "baseline_hr": 70,       # 기본 심박수 (BPM)
                    "hrv_params": {
                        "SDNN": 65.0,        # SDNN (ms)
                        "RMSSD": 42.0,       # RMSSD (ms)
                        "pNN50": 23.0,       # pNN50 (%)
                        "LF": 725.0,         # LF 파워 (ms²)
                        "HF": 975.0,         # HF 파워 (ms²)
                        "LF_HF_ratio": 0.7,  # LF/HF 비율
                        "SampEn": 1.55       # 샘플 엔트로피
                    }
                },
                "mild_anxiety": {
                    "description": "경미한 불안 패턴",
                    "duration": 300,         # 시나리오 지속 시간 (초)
                    "baseline_hr": 80,       # 기본 심박수 (BPM)
                    "hrv_params": {
                        "SDNN": 55.0,        # SDNN (ms)
                        "RMSSD": 35.0,       # RMSSD (ms)
                        "pNN50": 18.0,       # pNN50 (%)
                        "LF": 850.0,         # LF 파워 (ms²)
                        "HF": 800.0,         # HF 파워 (ms²)
                        "LF_HF_ratio": 1.06, # LF/HF 비율
                        "SampEn": 1.35       # 샘플 엔트로피
                    }
                },
                "moderate_anxiety": {
                    "description": "중등도 불안 패턴",
                    "duration": 300,         # 시나리오 지속 시간 (초)
                    "baseline_hr": 90,       # 기본 심박수 (BPM)
                    "hrv_params": {
                        "SDNN": 45.0,        # SDNN (ms)
                        "RMSSD": 28.0,       # RMSSD (ms)
                        "pNN50": 14.0,       # pNN50 (%)
                        "LF": 950.0,         # LF 파워 (ms²)
                        "HF": 650.0,         # HF 파워 (ms²)
                        "LF_HF_ratio": 1.46, # LF/HF 비율
                        "SampEn": 1.15       # 샘플 엔트로피
                    }
                },
                "severe_anxiety": {
                    "description": "심각한 불안 패턴",
                    "duration": 300,         # 시나리오 지속 시간 (초)
                    "baseline_hr": 105,      # 기본 심박수 (BPM)
                    "hrv_params": {
                        "SDNN": 35.0,        # SDNN (ms)
                        "RMSSD": 20.0,       # RMSSD (ms)
                        "pNN50": 8.0,        # pNN50 (%)
                        "LF": 1050.0,        # LF 파워 (ms²)
                        "HF": 500.0,         # HF 파워 (ms²)
                        "LF_HF_ratio": 2.10, # LF/HF 비율
                        "SampEn": 0.95       # 샘플 엔트로피
                    }
                },
                "panic": {
                    "description": "패닉 발작 패턴",
                    "duration": 300,         # 시나리오 지속 시간 (초)
                    "baseline_hr": 120,      # 기본 심박수 (BPM)
                    "hrv_params": {
                        "SDNN": 25.0,        # SDNN (ms)
                        "RMSSD": 15.0,       # RMSSD (ms)
                        "pNN50": 4.0,        # pNN50 (%)
                        "LF": 1200.0,        # LF 파워 (ms²)
                        "HF": 400.0,         # HF 파워 (ms²)
                        "LF_HF_ratio": 3.00, # LF/HF 비율
                        "SampEn": 0.75       # 샘플 엔트로피
                    }
                },
                "recovery": {
                    "description": "자극 후 회복 패턴",
                    "duration": 600,         # 시나리오 지속 시간 (초)
                    "baseline_hr": {
                        "start": 100,        # 시작 심박수 (BPM)
                        "end": 75            # 종료 심박수 (BPM)
                    },
                    "hrv_params": {
                        "start": {           # 시작 HRV 파라미터
                            "SDNN": 40.0,
                            "RMSSD": 25.0,
                            "pNN50": 10.0,
                            "LF": 900.0,
                            "HF": 600.0,
                            "LF_HF_ratio": 1.5,
                            "SampEn": 1.05
                        },
                        "end": {             # 종료 HRV 파라미터
                            "SDNN": 60.0,
                            "RMSSD": 38.0,
                            "pNN50": 20.0,
                            "LF": 750.0,
                            "HF": 850.0,
                            "LF_HF_ratio": 0.88,
                            "SampEn": 1.45
                        }
                    }
                }
            },
            "simulation_sequence": [
                {
                    "scenario": "normal",
                    "duration": 300          # 초
                },
                {
                    "scenario": "mild_anxiety",
                    "duration": 300
                },
                {
                    "scenario": "moderate_anxiety",
                    "duration": 300
                },
                {
                    "scenario": "severe_anxiety",
                    "duration": 300,
                    "auto_stimulation": True  # 자동 자극 활성화
                },
                {
                    "scenario": "recovery",
                    "duration": 600
                }
            ],
            "gui": {
                "width": 1200,               # 창 가로 크기
                "height": 800,               # 창 세로 크기
                "refresh_rate": 30,          # 화면 갱신 빈도 (Hz)
                "theme": "dark",             # 테마 (dark/light)
                "plot_history": 60,          # 플롯 히스토리 길이 (초)
                "layout": {
                    "ecg_plot_height": 3,    # ECG 플롯 상대적 높이
                    "hrv_plot_height": 2,    # HRV 플롯 상대적 높이
                    "anxiety_plot_height": 2, # 불안 플롯 상대적 높이
                    "stim_plot_height": 1    # 자극 플롯 상대적 높이
                }
            }
        }
        
        # 설정 파일이 존재하면 로드
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    
                # 깊은 병합 수행
                self._deep_update(default_config, loaded_config)
                    
            except Exception as e:
                print(f"Error loading config file {self.config_file}: {e}")
                print("Using default configuration")
                
        else:
            print(f"Config file {self.config_file} not found. Using default configuration")
            
            # 기본 설정 파일 저장
            try:
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                    
                print(f"Default configuration saved to {self.config_file}")
                
            except Exception as e:
                print(f"Error saving default configuration: {e}")
                
        return default_config
        
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        딕셔너리 깊은 병합
        
        Args:
            target: 대상 딕셔너리
            source: 소스 딕셔너리
        """
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v
                
    def save(self, filepath: Optional[str] = None) -> bool:
        """
        설정 저장
        
        Args:
            filepath: 저장 경로 (None인 경우 로드한 경로 사용)
            
        Returns:
            성공 여부
        """
        save_path = filepath or self.config_file
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            print(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
            
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        설정값 가져오기
        
        Args:
            section: 설정 섹션
            key: 설정 키 (None인 경우 섹션 전체 반환)
            default: 기본값
            
        Returns:
            설정값
        """
        if section not in self.config:
            return default
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key, default)
        
    def set(self, section: str, key: str, value: Any) -> None:
        """
        설정값 설정
        
        Args:
            section: 설정 섹션
            key: 설정 키
            value: 설정값
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
        
    def get_scenario(self, scenario_name: str) -> Dict:
        """
        시나리오 설정 가져오기
        
        Args:
            scenario_name: 시나리오 이름
            
        Returns:
            시나리오 설정 딕셔너리
        """
        scenarios = self.get("scenarios", {})
        return scenarios.get(scenario_name, {})
        
    def get_simulation_sequence(self) -> List[Dict]:
        """
        시뮬레이션 시퀀스 가져오기
        
        Returns:
            시뮬레이션 시퀀스 리스트
        """
        return self.get("simulation_sequence", [])
        
    def set_simulation_sequence(self, sequence: List[Dict]) -> None:
        """
        시뮬레이션 시퀀스 설정
        
        Args:
            sequence: 시뮬레이션 시퀀스 리스트
        """
        self.set("simulation", "simulation_sequence", sequence)
        

# 기본 인스턴스 생성 (싱글톤 패턴)
config = SimulationConfig()


if __name__ == "__main__":
    # 설정 로드 테스트
    import pprint
    
    print("불안장애 예방 시스템 시뮬레이션 설정")
    print(f"설정 파일: {config.config_file}")
    print("\n시뮬레이션 시퀀스:")
    pprint.pprint(config.get_simulation_sequence())
    
    print("\n시나리오 목록:")
    for scenario_name in config.get("scenarios", {}).keys():
        print(f"- {scenario_name}: {config.get_scenario(scenario_name).get('description', '')}")
