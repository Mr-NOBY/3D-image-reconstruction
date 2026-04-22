/**
 * 3D Reconstruction Viewer — Three.js Engine (ES Module)
 * Interactive point cloud viewer with PLY loading and orbit controls.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

// ─── State ───
let scene, camera, renderer, controls, pointCloud;
let autoRotate = false;
let frameCount = 0;
let lastFpsTime = performance.now();

// ─── DOM ───
const canvas = document.getElementById("viewer-canvas");
const loading = document.getElementById("loading");
const pointCountEl = document.getElementById("point-count");
const fpsCounterEl = document.getElementById("fps-counter");
const pointSizeSlider = document.getElementById("point-size");
const sizeValueEl = document.getElementById("size-value");
const fileDrop = document.getElementById("file-drop");
const fileInput = document.getElementById("file-input");

// ─── Init Scene ───
function init() {
    // Renderer — preserveDrawingBuffer needed for screenshots
    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x0a0a0f, 1);

    // Scene
    scene = new THREE.Scene();

    // Camera
    camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        10000
    );
    camera.position.set(0, 0, 5);

    // Controls — attach to renderer.domElement (the canvas)
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.minDistance = 0.1;
    controls.maxDistance = 1000;

    // Ambient light
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // Grid helper
    const grid = new THREE.GridHelper(10, 20, 0x222240, 0x15152a);
    grid.material.opacity = 0.3;
    grid.material.transparent = true;
    scene.add(grid);

    // Axes helper
    const axes = new THREE.AxesHelper(1);
    scene.add(axes);

    // Events
    window.addEventListener("resize", onResize);
    setupControls();

    // Start render loop
    animate();
}

// ─── Animation Loop ───
function animate() {
    requestAnimationFrame(animate);

    if (autoRotate && pointCloud) {
        pointCloud.rotation.y += 0.003;
    }

    controls.update();
    renderer.render(scene, camera);

    // FPS counter
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        fpsCounterEl.textContent = frameCount + " FPS";
        frameCount = 0;
        lastFpsTime = now;
    }
}

// ─── Load PLY ───
function loadPLY(source) {
    loading.classList.remove("hidden");

    const loader = new PLYLoader();

    const onLoad = function (geometry) {
        // Remove old point cloud
        if (pointCloud) {
            scene.remove(pointCloud);
            pointCloud.geometry.dispose();
            pointCloud.material.dispose();
        }

        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        // Center the geometry
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        // Scale to fit view
        const radius = geometry.boundingSphere.radius || 1;
        const scale = 3 / radius;
        geometry.scale(scale, scale, scale);

        // Material
        const hasColors = geometry.hasAttribute("color");
        const material = new THREE.PointsMaterial({
            size: parseFloat(pointSizeSlider.value) * 0.01,
            vertexColors: hasColors,
            color: hasColors ? 0xffffff : 0x6c5ce7,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
        });

        pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Update UI
        const count = geometry.attributes.position.count;
        pointCountEl.textContent = count.toLocaleString() + " points";

        // Fit camera
        camera.position.set(0, 1, 4);
        controls.target.set(0, 0, 0);
        controls.update();

        loading.classList.add("hidden");
        console.log(`Loaded ${count} points`);
    };

    const onError = function (err) {
        console.error("Error loading PLY:", err);
        loading.classList.add("hidden");
        alert("Failed to load PLY file. Check the console for details.");
    };

    if (source instanceof ArrayBuffer) {
        try {
            const geometry = loader.parse(source);
            onLoad(geometry);
        } catch (e) {
            onError(e);
        }
    } else {
        loader.load(source, onLoad, undefined, onError);
    }
}

// ─── File Input ───
function setupControls() {
    // --- File upload: click on drop zone opens file picker ---
    fileDrop.addEventListener("click", (e) => {
        // Don't trigger if they clicked the input itself
        if (e.target === fileInput) return;
        fileInput.click();
    });

    // --- File upload: drag & drop ---
    fileDrop.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileDrop.classList.add("drag-over");
    });

    fileDrop.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileDrop.classList.remove("drag-over");
    });

    fileDrop.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileDrop.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && file.name.toLowerCase().endsWith(".ply")) {
            console.log("Dropped file:", file.name, file.size, "bytes");
            readFile(file);
        } else {
            alert("Please drop a .ply file");
        }
    });

    // --- File upload: input change ---
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log("Selected file:", file.name, file.size, "bytes");
            readFile(file);
        }
    });

    // --- Point size slider ---
    pointSizeSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        sizeValueEl.textContent = val.toFixed(1);
        if (pointCloud) {
            pointCloud.material.size = val * 0.01;
            pointCloud.material.needsUpdate = true;
        }
    });

    // --- Background buttons ---
    document.querySelectorAll("[data-bg]").forEach((btn) => {
        btn.addEventListener("click", () => {
            document
                .querySelectorAll("[data-bg]")
                .forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const bg = btn.dataset.bg;
            if (bg === "dark") renderer.setClearColor(0x0a0a0f, 1);
            else if (bg === "light") renderer.setClearColor(0xe8e8f0, 1);
            else renderer.setClearColor(0x1a1a2e, 1);
        });
    });

    // --- View buttons ---
    document.getElementById("btn-front").addEventListener("click", () => {
        camera.position.set(0, 0, 5);
        controls.target.set(0, 0, 0);
        controls.update();
    });
    document.getElementById("btn-top").addEventListener("click", () => {
        camera.position.set(0, 5, 0.01);
        controls.target.set(0, 0, 0);
        controls.update();
    });
    document.getElementById("btn-side").addEventListener("click", () => {
        camera.position.set(5, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
    });
    document.getElementById("btn-reset").addEventListener("click", () => {
        camera.position.set(0, 1, 4);
        controls.target.set(0, 0, 0);
        if (pointCloud) pointCloud.rotation.set(0, 0, 0);
        controls.update();
    });

    // --- Screenshot ---
    document.getElementById("btn-screenshot").addEventListener("click", () => {
        renderer.render(scene, camera);
        const link = document.createElement("a");
        link.download = "reconstruction_screenshot.png";
        link.href = canvas.toDataURL("image/png");
        link.click();
    });

    // --- Auto rotate ---
    const rotateBtn = document.getElementById("btn-auto-rotate");
    rotateBtn.addEventListener("click", () => {
        autoRotate = !autoRotate;
        rotateBtn.classList.toggle("active", autoRotate);
    });
}

function readFile(file) {
    loading.classList.remove("hidden");
    const reader = new FileReader();
    reader.onload = (e) => {
        console.log("File read complete, loading into Three.js...");
        loadPLY(e.target.result);
    };
    reader.onerror = () => {
        console.error("FileReader error");
        loading.classList.add("hidden");
    };
    reader.readAsArrayBuffer(file);
}

// ─── Resize ───
function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ─── Start ───
init();
console.log("3D Reconstruction Viewer initialized");
