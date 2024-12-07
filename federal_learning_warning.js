function displayWarningDevices(warningDevices) {
    const warningContainer = document.createElement('div');
    warningContainer.className = 'warning-container';

    const warningTitle = document.createElement('div');
    warningTitle.className = 'warning-title';
    warningTitle.textContent = '损毁预警机器ID：';
    warningContainer.appendChild(warningTitle);

    const deviceList = document.createElement('div');
    deviceList.className = 'device-list';

    warningDevices.forEach(deviceId => {
        const deviceElement = document.createElement('span');
        deviceElement.className = 'device-id';
        deviceElement.textContent = deviceId;
        deviceList.appendChild(deviceElement);
    });

    warningContainer.appendChild(deviceList);
    
    // 清除之前的预警显示
    const existingWarning = document.querySelector('.warning-container');
    if (existingWarning) {
        existingWarning.remove();
    }

    // 将新的预警添加到指定位置
    const targetElement = document.getElementById('warning-section');
    if (targetElement) {
        targetElement.appendChild(warningContainer);
    }
} 