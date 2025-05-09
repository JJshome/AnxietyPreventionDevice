#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG 센서 인터페이스 모듈

웨어러블 심전도 센서와의 통신을 위한 인터페이스 코드입니다.
"""

import time
import threading
import logging
from queue import Queue

# 블루투스 통신을 위한 라이브러리 임포트
try:
    import bluetooth
    HAVE_BLUETOOTH = True
except ImportError:
    HAVE_BLUETOOTH = False
    print("[!] PyBluez 라이브러리가 설치되지 않았습니다. 가상 ECG 센서만 사용 가능합니다.")
    
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ECGSensor")


class ECGSensorInterface:
    """
    ECG 센서 인터페이스 클래스
    """
    
    def __init__(self, use_virtual_sensor=False, sampling_rate=256):
        """
        초기화
        
        Args:
            use_virtual_sensor (bool): 가상 센서 사용 여부
            sampling_rate (int): 샘플링 레이트 (Hz)
        """
        self.use_virtual_sensor = use_virtual_sensor
        self.sampling_rate = sampling_rate
        self.socket = None
        self.connected = False
        self.device_address = None
        self.device_name = None
        
        # 센서 데이터 수신 로직
        self.receiving = False
        self.data_thread = None
        self.data_queue = Queue(maxsize=100)
        
        # 센서 상태
        self.battery_level = 100
        self.signal_quality = 100
        
        logger.info(f"ECG 센서 인터페이스 초기화 (샘플링 레이트: {sampling_rate}Hz)")
    
    def scan_devices(self, timeout=5):
        """
        주변 블루투스 ECG 센서 검색
        
        Args:
            timeout (int): 검색 시간(초)
            
        Returns:
            list: 검색된 센서 목록
        """
        if self.use_virtual_sensor:
            # 가상 센서 사용 시 1개의 가상 기기 리턴
            return [{
                "name": "Virtual ECG Sensor",
                "address": "00:00:00:00:00:00"
            }]
        
        if not HAVE_BLUETOOTH:
            logger.error("BlueZ 라이브러리가 설치되지 않아 블루투스 검색을 수행할 수 없습니다.")
            return []
        
        logger.info(f"Bluetooth ECG 센서 검색 중... ({timeout}초)")
        
        try:
            # 블루투스 검색 수행
            nearby_devices = bluetooth.discover_devices(duration=timeout, lookup_names=True)
            
            # 검색된 기기 필터링
            ecg_devices = []
            for addr, name in nearby_devices:
                # ECG 센서 기기 필터링 (이름에 ECG, HRM, 또는 MyBeat가 포함된 경우만 추출)
                if "ECG" in name or "HRM" in name or "MyBeat" in name or "Heart" in name:
                    ecg_devices.append({
                        "name": name,
                        "address": addr
                    })
                    logger.info(f"  발견된 ECG 센서: {name} ({addr})")
            
            return ecg_devices
        except Exception as e:
            logger.error(f"Bluetooth 검색 오류: {e}")
            return []
    
    def connect(self, device_address):
        """
        ECG 센서에 연결
        
        Args:
            device_address (str): 센서 주소
            
        Returns:
            bool: 연결 성공 여부
        """
        if self.connected:
            logger.warning("이미 센서에 연결되어 있습니다. disconnect()를 먼저 호출하세요.")
            return False
        
        self.device_address = device_address
        
        if self.use_virtual_sensor:
            logger.info("가상 ECG 센서에 연결되었습니다.")
            self.device_name = "Virtual ECG Sensor"
            self.connected = True
            self.battery_level = 85
            return True
        
        if not HAVE_BLUETOOTH:
            logger.error("BlueZ 라이브러리가 설치되지 않아 블루투스 연결을 수행할 수 없습니다.")
            return False
        
        try:
            # RFCOMM 서비스 포트 검색 (실제 센서에 맞게 조정 필요)
            port = 1  # 기본 RFCOMM 포트
            
            # 소켓 생성 및 연결
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.connect((device_address, port))
            self.socket.settimeout(1.0)  # 1초 타임아웃 설정
            
            # 연결 확인
            self.device_name = "Unknown ECG Sensor"
            try:
                # 기기 정보 요청 명령 전송
                self._send_command(b"GET_INFO\r\n")
                response = self._read_response()
                if response and b"name" in response:
                    self.device_name = response.decode('utf-8', errors='ignore').split(':')[1].strip()
            except Exception as e:
                logger.warning(f"  기기 정보 읽기 실패: {e}")
            
            self.connected = True
            logger.info(f"ECG 센서에 연결되었습니다: {self.device_name}")
            
            # 배터리 정보 가져오기
            self.get_battery_level()
            
            return True
        except Exception as e:
            logger.error(f"ECG 센서 연결 오류: {e}")
            self.socket = None
            return False
    
    def disconnect(self):
        """
        ECG 센서와 연결 해제
        
        Returns:
            bool: 연결 해제 성공 여부
        """
        if not self.connected:
            logger.warning("ECG 센서에 연결되어 있지 않습니다.")
            return True
        
        if self.receiving:
            self.stop_data_stream()
        
        if self.use_virtual_sensor:
            self.connected = False
            self.device_address = None
            self.device_name = None
            logger.info("가상 ECG 센서 연결이 해제되었습니다.")
            return True
        
        if self.socket:
            try:
                self.socket.close()
                self.socket = None
                self.connected = False
                self.device_address = None
                self.device_name = None
                logger.info("ECG 센서 연결이 해제되었습니다.")
                return True
            except Exception as e:
                logger.error(f"ECG 센서 연결 해제 오류: {e}")
                return False
        else:
            self.connected = False
            return True
    
    def _send_command(self, command):
        """
        센서에 명령 전송
        
        Args:
            command (bytes): 전송할 명령 바이트
            
        Returns:
            bool: 전송 성공 여부
        """
        if self.use_virtual_sensor:
            return True
        
        if not self.connected or not self.socket:
            logger.error("센서에 연결되어 있지 않습니다.")
            return False
        
        try:
            self.socket.send(command)
            return True
        except Exception as e:
            logger.error(f"명령 전송 오류: {e}")
            return False
    
    def _read_response(self, timeout=1.0, buffer_size=1024):
        """
        센서로부터 응답 읽기
        
        Args:
            timeout (float): 읽기 시간 제한(초)
            buffer_size (int): 버퍼 크기
            
        Returns:
            bytes: 읽은 데이터
        """
        if self.use_virtual_sensor:
            return b"OK"  # 가상 센서의 경우 항상 OK 반환
        
        if not self.connected or not self.socket:
            logger.error("센서에 연결되어 있지 않습니다.")
            return None
        
        try:
            # 타임아웃 설정
            self.socket.settimeout(timeout)
            
            # 데이터 읽기
            response = self.socket.recv(buffer_size)
            return response
        except bluetooth.btcommon.BluetoothError as e:
            if "timed out" in str(e).lower():
                return None  # 타임아웃은 오류로 처리하지 않음
            else:
                logger.error(f"응답 읽기 오류: {e}")
                return None
        except Exception as e:
            logger.error(f"응답 읽기 오류: {e}")
            return None
    
    def get_battery_level(self):
        """
        배터리 잔량 조회
        
        Returns:
            int: 배터리 잔량 (%)
        """
        if self.use_virtual_sensor:
            # 가상 센서 배터리 레벨 감소
            self.battery_level = max(0, self.battery_level - 0.1)
            return int(self.battery_level)
        
        if not self.connected:
            logger.error("센서에 연결되어 있지 않습니다.")
            return 0
        
        try:
            # 배터리 레벨 요청 명령 전송
            self._send_command(b"GET_BATTERY\r\n")
            response = self._read_response()
            
            if response and b"BATTERY" in response:
                try:
                    # 배터리 값 파싱
                    battery_level = int(response.decode('utf-8').split(':')[1].strip())
                    self.battery_level = battery_level
                    return battery_level
                except (ValueError, IndexError):
                    logger.error("배터리 레벨 파싱 오류")
            
            # 응답이 없거나 파싱 불가 시 추정값 사용
            return self.battery_level
        except Exception as e:
            logger.error(f"배터리 레벨 조회 오류: {e}")
            return self.battery_level
    
    def get_signal_quality(self):
        """
        신호 품질 조회
        
        Returns:
            int: 신호 품질 (%)
        """
        if self.use_virtual_sensor:
            return 95  # 가상 센서는 항상 양호한 신호
        
        if not self.connected:
            logger.error("센서에 연결되어 있지 않습니다.")
            return 0
        
        try:
            # 신호 품질 요청 명령 전송
            self._send_command(b"GET_SIGNAL_QUALITY\r\n")
            response = self._read_response()
            
            if response and b"QUALITY" in response:
                try:
                    # 신호 품질 값 파싱
                    quality = int(response.decode('utf-8').split(':')[1].strip())
                    self.signal_quality = quality
                    return quality
                except (ValueError, IndexError):
                    logger.error("신호 품질 파싱 오류")
            
            # 응답이 없거나 파싱 불가 시 추정값 사용
            return self.signal_quality
        except Exception as e:
            logger.error(f"신호 품질 조회 오류: {e}")
            return self.signal_quality
    
    def start_data_stream(self):
        """
        ECG 데이터 스트림 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        if not self.connected:
            logger.error("센서에 연결되어 있지 않습니다.")
            return False
        
        if self.receiving:
            logger.warning("이미 데이터 스트림이 실행 중입니다.")
            return True
        
        # 데이터 큐 초기화
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        
        # 가상 센서 사용 경우
        if self.use_virtual_sensor:
            self.receiving = True
            self.data_thread = threading.Thread(target=self._virtual_data_stream)
            self.data_thread.daemon = True
            self.data_thread.start()
            logger.info("가상 ECG 데이터 스트림이 시작되었습니다.")
            return True
        
        # 실제 센서에 데이터 스트림 시작 명령 전송
        success = self._send_command(b"START_STREAM\r\n")
        if not success:
            logger.error("데이터 스트림 시작 명령 전송 실패")
            return False
        
        # 응답 확인
        response = self._read_response()
        if not response or b"OK" not in response:
            logger.error(f"데이터 스트림 시작 실패: {response}")
            return False
        
        # 데이터 스트림 수신 스레드 시작
        self.receiving = True
        self.data_thread = threading.Thread(target=self._receive_data_stream)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        logger.info("ECG 데이터 스트림이 시작되었습니다.")
        return True
    
    def stop_data_stream(self):
        """
        ECG 데이터 스트림 중지
        
        Returns:
            bool: 중지 성공 여부
        """
        if not self.receiving:
            logger.warning("데이터 스트림이 실행 중이지 않습니다.")
            return True
        
        # 가상 센서 사용 경우
        if self.use_virtual_sensor:
            self.receiving = False
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=1.0)
            self.data_thread = None
            logger.info("가상 ECG 데이터 스트림이 중지되었습니다.")
            return True
        
        # 실제 센서에 데이터 스트림 중지 명령 전송
        success = self._send_command(b"STOP_STREAM\r\n")
        if not success:
            logger.error("데이터 스트림 중지 명령 전송 실패")
            return False
        
        # 응답 확인
        response = self._read_response()
        if not response or b"OK" not in response:
            logger.error(f"데이터 스트림 중지 실패: {response}")
            return False
        
        # 스레드 중지
        self.receiving = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
        self.data_thread = None
        
        logger.info("ECG 데이터 스트림이 중지되었습니다.")
        return True
    
    def _receive_data_stream(self):
        """
        실제 센서로부터 데이터 스트림 수신 (스레드 함수)
        """
        import numpy as np
        
        buffer_size = 1024
        self.socket.settimeout(0.1)  # 빠른 응답을 위해 타임아웃 설정
        
        logger.info("데이터 스트림 수신 스레드 시작")
        
        while self.receiving:
            try:
                # 데이터 읽기
                data = self.socket.recv(buffer_size)
                
                if not data:
                    continue
                
                # 패킷 처리 (실제 프로토콜에 맞게 조정 필요)
                # 예시: 간단한 패킷 형식 (헤더: 1바이트, 데이터: 2바이트 * N)
                if len(data) < 3 or data[0] != 0x55:  # 헤더 확인
                    continue
                
                # 데이터 추출 및 변환
                values = []
                for i in range(1, len(data), 2):
                    if i + 1 < len(data):
                        # 2바이트 값을 하나의 샘플로 변환
                        value = (data[i] << 8) | data[i+1]
                        values.append(value)
                
                # 값 변환 및 큐에 추가
                if values:
                    # ECG 값 변환 (실제 센서에 맞게 조정 필요)
                    ecg_values = np.array(values) * 3.3 / 4096  # 예: 12비트 ADC, 3.3V 레퍼런스
                    
                    # 데이터 큐에 추가
                    try:
                        self.data_queue.put(ecg_values, block=False)
                    except Queue.Full:
                        # 큐가 가득찬 경우 가장 오래된 항목 제거 후 추가
                        self.data_queue.get()
                        self.data_queue.put(ecg_values, block=False)
            
            except bluetooth.btcommon.BluetoothError as e:
                if "timed out" in str(e).lower():
                    continue  # 타임아웃은 정상으로 처리
                else:
                    logger.error(f"데이터 스트림 수신 오류: {e}")
                    break
            
            except Exception as e:
                logger.error(f"데이터 스트림 처리 오류: {e}")
                time.sleep(0.1)  # 오류 발생 시 잠시 대기
        
        logger.info("데이터 스트림 수신 스레드 종료")
    
    def _virtual_data_stream(self):
        """
        가상 ECG 데이터 스트림 생성 (스레드 함수)
        """
        import numpy as np
        from scipy import signal
        
        logger.info("가상 ECG 데이터 스트림 스레드 시작")
        
        # 가상 ECG 신호 생성 파라미터
        sampling_rate = self.sampling_rate
        duration = 10  # 10초 단위로 데이터 생성
        t = np.arange(0, duration, 1/sampling_rate)
        
        # 기본 심박수 및 변화량
        base_hr = 60  # 기본 심박수 (bpm)
        hr_var = 10   # 심박수 변화량 (bpm)
        
        while self.receiving:
            try:
                # 현재 심박수 설정 (랜덤 변화)
                current_hr = base_hr + np.random.uniform(-hr_var, hr_var)
                period = 60 / current_hr  # 심박수를 주기로 변환
                
                # ECG 신호 생성
                ecg_signal = self._generate_synthetic_ecg(t, period)
                
                # 잡음 추가
                noise = np.random.normal(0, 0.05, len(ecg_signal))
                ecg_signal = ecg_signal + noise
                
                # 데이터 분할 전송 (실제 센서처럼 패킷 단위 전송 시뮬레이션)
                chunk_size = int(sampling_rate * 0.2)  # 0.2초 단위로 분할
                for i in range(0, len(ecg_signal), chunk_size):
                    if not self.receiving:
                        break
                    
                    chunk = ecg_signal[i:i+chunk_size]
                    
                    # 데이터 큐에 추가
                    try:
                        self.data_queue.put(chunk, block=False)
                    except Queue.Full:
                        # 큐가 가득찬 경우 가장 오래된 항목 제거 후 추가
                        self.data_queue.get()
                        self.data_queue.put(chunk, block=False)
                    
                    # 실제 하드웨어 시간 시뮬레이션
                    time.sleep(len(chunk) / sampling_rate)
                
            except Exception as e:
                logger.error(f"가상 ECG 데이터 생성 오류: {e}")
                time.sleep(0.5)  # 오류 발생 시 잠시 대기
        
        logger.info("가상 ECG 데이터 스트림 스레드 종료")
    
    def _generate_synthetic_ecg(self, t, period=1.0):
        """
        가상 ECG 신호 생성
        
        Args:
            t (numpy.ndarray): 시간 배열
            period (float): 심박 주기(초)
            
        Returns:
            numpy.ndarray: 생성된 ECG 신호
        """
        import numpy as np
        
        # ECG 파형 파라미터
        p_width = 0.08  # P-wave 폭
        p_amp = 0.15    # P-wave 진폭
        qrs_width = 0.05  # QRS 폭
        qrs_amp = 1.0     # QRS 진폭
        t_width = 0.1    # T-wave 폭
        t_amp = 0.3      # T-wave 진폭
        
        # 파형 생성 함수
        def gaussian(x, amp, mean, std):
            return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        # 시간 정규화 (0~1 범위의 심장 주기 위치)
        t_norm = (t % period) / period
        
        # 각 파형 생성
        p_wave = gaussian(t_norm, p_amp, 0.16, p_width)
        q_wave = gaussian(t_norm, -0.15 * qrs_amp, 0.3, 0.02)
        r_wave = gaussian(t_norm, qrs_amp, 0.33, 0.02)
        s_wave = gaussian(t_norm, -0.4 * qrs_amp, 0.36, 0.02)
        t_wave = gaussian(t_norm, t_amp, 0.5, t_width)
        
        # 합성 ECG 신호
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        
        # 기준선 미세 변동 추가
        baseline = 0.05 * np.sin(2 * np.pi * 0.1 * t)  # 호흡에 의한 기준선 변동
        
        return ecg + baseline
    
    def get_data(self, block=True, timeout=1.0):
        """
        ECG 데이터 가져오기
        
        Args:
            block (bool): 데이터가 없을 경우 대기 여부
            timeout (float): 대기 시간(초)
            
        Returns:
            numpy.ndarray: ECG 데이터 또는 None
        """
        if not self.receiving:
            logger.warning("데이터 스트림이 실행 중이지 않습니다.")
            return None
        
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except Queue.Empty:
            return None
        except Exception as e:
            logger.error(f"데이터 가져오기 오류: {e}")
            return None
    
    def set_sampling_rate(self, sampling_rate):
        """
        샘플링 레이트 설정
        
        Args:
            sampling_rate (int): 설정할 샘플링 레이트(Hz)
            
        Returns:
            bool: 설정 성공 여부
        """
        if self.receiving:
            logger.warning("데이터 스트림이 실행 중입니다. 변경하려면 먼저 중지하세요.")
            return False
        
        if self.use_virtual_sensor:
            self.sampling_rate = sampling_rate
            logger.info(f"가상 센서 샘플링 레이트 변경: {sampling_rate}Hz")
            return True
        
        # 실제 센서 샘플링 레이트 변경 명령 전송
        success = self._send_command(f"SET_SAMPLING_RATE {sampling_rate}\r\n".encode())
        if not success:
            logger.error("샘플링 레이트 변경 명령 전송 실패")
            return False
        
        # 응답 확인
        response = self._read_response()
        if not response or b"OK" not in response:
            logger.error(f"샘플링 레이트 변경 실패: {response}")
            return False
        
        self.sampling_rate = sampling_rate
        logger.info(f"센서 샘플링 레이트 변경: {sampling_rate}Hz")
        return True
