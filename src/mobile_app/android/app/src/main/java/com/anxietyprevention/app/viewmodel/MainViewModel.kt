package com.anxietyprevention.app.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.anxietyprevention.app.data.DeviceManager
import com.anxietyprevention.app.data.StimulationManager
import com.anxietyprevention.app.data.model.AnxietyLevel
import com.anxietyprevention.app.data.repository.DeviceRepository
import com.anxietyprevention.app.data.repository.HrvRepository
import com.anxietyprevention.app.data.repository.SessionRepository
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val deviceRepository = DeviceRepository(application)
    private val hrvRepository = HrvRepository(application)
    private val sessionRepository = SessionRepository(application)
    
    private val deviceManager = DeviceManager(application, deviceRepository)
    private val stimulationManager = StimulationManager(application, sessionRepository)
    
    private val _uiEvents = MutableSharedFlow<UiEvent>()
    val uiEvents: SharedFlow<UiEvent> = _uiEvents.asSharedFlow()
    
    init {
        viewModelScope.launch {
            deviceManager.deviceEvents.collect { event ->
                when (event) {
                    is DeviceManager.DeviceEvent.DeviceConnected -> 
                        _uiEvents.emit(UiEvent.DeviceConnected(event.deviceName))
                    is DeviceManager.DeviceEvent.DeviceDisconnected -> 
                        _uiEvents.emit(UiEvent.DeviceDisconnected(event.deviceName))
                    is DeviceManager.DeviceEvent.ScanStarted -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Scanning for devices..."))
                    is DeviceManager.DeviceEvent.ScanFinished -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Scan completed"))
                    is DeviceManager.DeviceEvent.DeviceFound -> 
                        _uiEvents.emit(UiEvent.ShowToast("Found device: ${event.deviceName}"))
                    is DeviceManager.DeviceEvent.Error -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Error: ${event.message}"))
                }
            }
        }
        
        viewModelScope.launch {
            hrvRepository.anxietyLevelUpdates.collect { anxietyLevel ->
                if (anxietyLevel.level >= 2) {  // Alert for high levels of anxiety (2-3)
                    _uiEvents.emit(UiEvent.AnxietyAlertDetected(anxietyLevel.level))
                }
            }
        }
        
        viewModelScope.launch {
            stimulationManager.stimulationEvents.collect { event ->
                when (event) {
                    is StimulationManager.StimulationEvent.Started -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Stimulation started"))
                    is StimulationManager.StimulationEvent.Stopped -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Stimulation stopped"))
                    is StimulationManager.StimulationEvent.Error -> 
                        _uiEvents.emit(UiEvent.ShowSnackbar("Stimulation error: ${event.message}"))
                }
            }
        }
    }
    
    fun initializeDeviceManager() {
        deviceManager.initialize()
    }
    
    fun startScan() {
        viewModelScope.launch {
            deviceManager.startScan()
        }
    }
    
    fun connectToDevice(deviceAddress: String) {
        viewModelScope.launch {
            deviceManager.connectToDevice(deviceAddress)
        }
    }
    
    fun disconnectFromDevice(deviceAddress: String) {
        viewModelScope.launch {
            deviceManager.disconnectFromDevice(deviceAddress)
        }
    }
    
    fun startStimulation(anxietyLevel: Int) {
        viewModelScope.launch {
            stimulationManager.startStimulation(AnxietyLevel(anxietyLevel))
        }
    }
    
    fun stopStimulation() {
        viewModelScope.launch {
            stimulationManager.stopStimulation()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        deviceManager.cleanup()
        stimulationManager.cleanup()
    }
    
    sealed class UiEvent {
        data class ShowToast(val message: String) : UiEvent()
        data class ShowSnackbar(val message: String) : UiEvent()
        data class NavigateTo(val destinationId: Int) : UiEvent()
        data class DeviceConnected(val deviceName: String) : UiEvent()
        data class DeviceDisconnected(val deviceName: String) : UiEvent()
        data class AnxietyAlertDetected(val level: Int) : UiEvent()
    }
}
