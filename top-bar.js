// top-bar.js
function expandTopBar(button) {
    const topBar = document.querySelector('.top-bar');
    topBar.classList.add('expanded');
    
    // 检查是否已经存在模糊层
    let blurLayer = document.querySelector('.blur-layer');
    if (!blurLayer) {
        // 创建模糊层
        blurLayer = document.createElement('div');
        blurLayer.className = 'blur-layer';
        document.body.appendChild(blurLayer); // 添加模糊层到文档中

        // 设置初始透明度
        blurLayer.style.opacity = '0';
        setTimeout(() => {
            blurLayer.style.opacity = '1'; // 逐渐显示模糊层
        }, 0);
    }
}

document.querySelectorAll('.nav-buttons > div').forEach(navItem => {
    const button = navItem.querySelector('.nav-button');
    const subMenu = navItem.querySelector('.sub-menu');

    let isMouseOverButton = false;
    let isMouseOverSubMenu = false;

    // 鼠标悬停在按钮时，显示子菜单
    button.addEventListener('mouseenter', () => {
        isMouseOverButton = true;
        subMenu.style.display = 'block'; // 显示子菜单
        setTimeout(() => {
            subMenu.classList.add('show');  // 添加 show 类
        }, 0.2); // 修改为200毫秒
    });

    // 鼠标离开按钮时，设置标志并检查隐藏
    button.addEventListener('mouseleave', () => {
        isMouseOverButton = false;
        setTimeout(checkHideSubMenu, 200); // 增加延迟
    });

    // 鼠标悬停在子菜单时，保持子菜单显示
    subMenu.addEventListener('mouseenter', () => {
        isMouseOverSubMenu = true;  // 设置标志
        subMenu.classList.add('show');  // 保持子菜单显示
    });

    // 鼠标离开子菜单时，设置标志并检查隐藏
    subMenu.addEventListener('mouseleave', () => {
        isMouseOverSubMenu = false;
        setTimeout(checkHideSubMenu, 200); // 增加延迟
    });

    // 检查是否需要隐藏子菜单
    function checkHideSubMenu() {
        // 只有当鼠标同时离开按钮和子菜单时，才隐藏子菜单
        if (!isMouseOverButton && !isMouseOverSubMenu) {
            subMenu.classList.remove('show');  // 隐藏子菜单
            setTimeout(() => {
                subMenu.style.display = 'none'; // 隐藏子菜单
            }, 200); // 等待过渡完成后再隐藏
        }
    }
});

// 鼠标离开 top-bar 时，收回 top-bar
const topBar = document.querySelector('.top-bar');
topBar.addEventListener('mouseleave', () => {
    topBar.classList.remove('expanded'); // 收回 top-bar
    const blurLayer = document.querySelector('.blur-layer');
    if (blurLayer) {
        blurLayer.style.opacity = '0'; // 逐渐隐藏模糊层
        setTimeout(() => {
            blurLayer.remove(); // 移除模糊层
        }, 200); // 等待200毫秒后移除
    }
    const allSubMenus = document.querySelectorAll('.sub-menu');
    allSubMenus.forEach(menu => {
        menu.classList.remove('show'); // 隐藏所有子菜单
        setTimeout(() => {
            menu.style.display = 'none'; // 隐藏所有子菜单
        }, 200); // 等待过渡完成后再隐藏
    });
});
