// 初始化Three.js场景
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('three-container').appendChild(renderer.domElement);

// 设置相机位置
camera.position.set(0, 1, 5);

// 添加光源
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(5, 5, 5);
scene.add(directionalLight);

// 添加气泡
function createBubble() {
    const bubbleGeometry = new THREE.SphereGeometry(0.5, 32, 32);
    const bubbleMaterial = new THREE.MeshStandardMaterial({ color: 0x9B6DFF, transparent: true, opacity: 0.3 });
    const bubble = new THREE.Mesh(bubbleGeometry, bubbleMaterial);
    bubble.position.set(Math.random() * 10 - 5, Math.random() * 10 - 5, Math.random() * -10);
    scene.add(bubble);
    return bubble;
}

const bubbles = Array.from({ length: 10 }, createBubble);

// 动画循环
function animate() {
    requestAnimationFrame(animate);

    // 气泡动画
    bubbles.forEach(bubble => {
        bubble.position.y += 0.01;
        if (bubble.position.y > 5) {
            bubble.position.y = -5;
        }
    });

    renderer.render(scene, camera);
}

animate();

// 监听窗口大小变化
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}); 