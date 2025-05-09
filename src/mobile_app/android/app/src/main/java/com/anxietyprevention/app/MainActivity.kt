package com.anxietyprevention.app

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.anxietyprevention.app.bluetooth.BluetoothService
import com.anxietyprevention.app.databinding.ActivityMainBinding
import com.anxietyprevention.app.viewmodel.MainViewModel
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var appBarConfiguration: AppBarConfiguration
    private val viewModel: MainViewModel by viewModels()
    
    private var bluetoothAdapter: BluetoothAdapter? = null
    
    // Bluetooth permission request
    private val requestBlePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (allGranted) {
            startBluetoothService()
        } else {
            showBluetoothPermissionDeniedDialog()
        }
    }
    
    // Bluetooth enable request
    private val requestEnableBluetooth = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            checkBluetoothPermissions()
        } else {
            showBluetoothDisabledDialog()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setSupportActionBar(binding.toolbar)
        
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.nav_dashboard, R.id.nav_devices, R.id.nav_settings
            ), binding.drawerLayout
        )
        setupActionBarWithNavController(navController, appBarConfiguration)
        binding.navView.setupWithNavController(navController)
        
        // Initialize Bluetooth adapter
        val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
        
        // Set up FloatingActionButton for scanning
        binding.fab.setOnClickListener { view ->
            if (isBluetoothEnabled()) {
                Snackbar.make(view, "Scanning for devices...", Snackbar.LENGTH_LONG).show()
                viewModel.startScan()
            } else {
                requestEnableBluetooth()
            }
        }
        
        // Observe ViewModel events
        observeViewModelEvents()
        
        // Check Bluetooth permissions and start service if possible
        checkBluetoothAvailability()
    }
    
    private fun observeViewModelEvents() {
        lifecycleScope.launch {
            viewModel.uiEvents.collect { event ->
                when (event) {
                    is MainViewModel.UiEvent.ShowToast -> 
                        Toast.makeText(this@MainActivity, event.message, Toast.LENGTH_SHORT).show()
                    is MainViewModel.UiEvent.ShowSnackbar -> 
                        Snackbar.make(binding.root, event.message, Snackbar.LENGTH_LONG).show()
                    is MainViewModel.UiEvent.NavigateTo -> 
                        findNavController(R.id.nav_host_fragment_content_main).navigate(event.destinationId)
                    is MainViewModel.UiEvent.DeviceConnected -> 
                        showDeviceConnectedDialog(event.deviceName)
                    is MainViewModel.UiEvent.DeviceDisconnected -> 
                        showDeviceDisconnectedDialog(event.deviceName)
                    is MainViewModel.UiEvent.AnxietyAlertDetected -> 
                        showAnxietyAlertDialog(event.level)
                }
            }
        }
    }
    
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                findNavController(R.id.nav_host_fragment_content_main).navigate(R.id.nav_settings)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }
    
    private fun checkBluetoothAvailability() {
        if (bluetoothAdapter == null) {
            // Device doesn't support Bluetooth
            showBluetoothUnavailableDialog()
            return
        }
        
        if (!isBluetoothEnabled()) {
            requestEnableBluetooth()
        } else {
            checkBluetoothPermissions()
        }
    }
    
    private fun isBluetoothEnabled(): Boolean {
        return bluetoothAdapter?.isEnabled == true
    }
    
    private fun requestEnableBluetooth() {
        val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
        requestEnableBluetooth.launch(enableBtIntent)
    }
    
    private fun checkBluetoothPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Android 12+ requires BLUETOOTH_SCAN and BLUETOOTH_CONNECT permissions
            val requiredPermissions = arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            )
            
            val hasPermissions = requiredPermissions.all {
                ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            }
            
            if (!hasPermissions) {
                requestBlePermissionLauncher.launch(requiredPermissions)
            } else {
                startBluetoothService()
            }
        } else {
            // For Android 11 and below, check LOCATION permissions
            val locationPermission = Manifest.permission.ACCESS_FINE_LOCATION
            
            if (ContextCompat.checkSelfPermission(this, locationPermission) != PackageManager.PERMISSION_GRANTED) {
                requestBlePermissionLauncher.launch(arrayOf(locationPermission))
            } else {
                startBluetoothService()
            }
        }
    }
    
    private fun startBluetoothService() {
        val serviceIntent = Intent(this, BluetoothService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }
        
        viewModel.initializeDeviceManager()
        Snackbar.make(binding.root, "Bluetooth service started", Snackbar.LENGTH_SHORT).show()
    }
    
    private fun showBluetoothUnavailableDialog() {
        AlertDialog.Builder(this)
            .setTitle("Bluetooth Unavailable")
            .setMessage("This device does not support Bluetooth, which is required for this app to function.")
            .setPositiveButton("OK") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }
    
    private fun showBluetoothDisabledDialog() {
        AlertDialog.Builder(this)
            .setTitle("Bluetooth Disabled")
            .setMessage("Bluetooth is required for this app to function. Would you like to enable it?")
            .setPositiveButton("Enable") { _, _ -> requestEnableBluetooth() }
            .setNegativeButton("Exit") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }
    
    private fun showBluetoothPermissionDeniedDialog() {
        AlertDialog.Builder(this)
            .setTitle("Permission Required")
            .setMessage("This app requires Bluetooth permissions to function correctly. Would you like to grant them?")
            .setPositiveButton("Grant") { _, _ -> checkBluetoothPermissions() }
            .setNegativeButton("Exit") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }
    
    private fun showDeviceConnectedDialog(deviceName: String) {
        Snackbar.make(binding.root, "Connected to $deviceName", Snackbar.LENGTH_LONG).show()
    }
    
    private fun showDeviceDisconnectedDialog(deviceName: String) {
        Snackbar.make(binding.root, "Disconnected from $deviceName", Snackbar.LENGTH_LONG).show()
    }
    
    private fun showAnxietyAlertDialog(level: Int) {
        val levelText = when (level) {
            1 -> "Moderate"
            2 -> "High"
            3 -> "Very High"
            else -> "Low"
        }
        
        AlertDialog.Builder(this)
            .setTitle("Anxiety Alert")
            .setMessage("Detected $levelText anxiety level. Would you like to start stimulation?")
            .setPositiveButton("Start Stimulation") { _, _ -> viewModel.startStimulation(level) }
            .setNegativeButton("Ignore") { dialog, _ -> dialog.dismiss() }
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Stop Bluetooth service
        stopService(Intent(this, BluetoothService::class.java))
    }
}
