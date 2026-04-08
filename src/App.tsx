import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./App.css";

type ProgressState = {
  stage: string;
  percent: number;
};

type MeshPayload = {
  vertices: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  stats: {
    gridSize: number;
    basisCount: number;
    occOrbitals: number;
    maxDensity: number;
  };
};

const GRID_PRESETS = [
  { label: "Preview (50^3)", value: 50 },
  { label: "Standard (80^3)", value: 80 },
  { label: "Heavy (120^3)", value: 120 },
];

function App() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const [fileName, setFileName] = useState<string | null>(null);
  const [fchkText, setFchkText] = useState<string | null>(null);
  const [gridSize, setGridSize] = useState(80);
  const [padding, setPadding] = useState(4);
  const [aValue, setAValue] = useState(1.1);
  const [isoValue, setIsoValue] = useState(1.0);
  const [colorMin, setColorMin] = useState(-0.04);
  const [colorMax, setColorMax] = useState(0.02);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<MeshPayload["stats"] | null>(null);

  const heavyRun = gridSize > 80;
  const memoryEstimateMb = useMemo(() => {
    const cells = gridSize * gridSize * gridSize;
    const bytes = cells * 4 * 4;
    return Math.round((bytes / (1024 * 1024)) * 10) / 10;
  }, [gridSize]);

  useEffect(() => {
    const workerUrl = new URL("./workers/iriWorker.ts", import.meta.url);
    workerUrl.searchParams.set("v", String(Date.now()));
    const worker = new Worker(workerUrl, { type: "module" });
    workerRef.current = worker;

    worker.onmessage = (event) => {
      const { type } = event.data;
      if (type === "progress") {
        setProgress({ stage: event.data.stage, percent: event.data.percent });
      }
      if (type === "error") {
        setBusy(false);
        setProgress(null);
        setError(event.data.message);
      }
      if (type === "result") {
        setBusy(false);
        setProgress(null);
        setError(null);
        const payload = event.data.payload as MeshPayload;
        setStats(payload.stats);
        updateMesh(payload);
      }
    };

    return () => worker.terminate();
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0e1116");
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    camera.position.set(0, 0, 80);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    const directional = new THREE.DirectionalLight(0xffffff, 0.9);
    directional.position.set(1, 1, 1);
    scene.add(ambient, directional);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;

    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current)
        return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      rendererRef.current.setSize(width, height);
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
    };

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    handleResize();
    animate();

    return () => {
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  const updateMesh = (payload: MeshPayload) => {
    if (!sceneRef.current) return;
    if (meshRef.current) {
      sceneRef.current.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      (meshRef.current.material as THREE.Material).dispose();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(payload.vertices, 3)
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(payload.colors, 3)
    );
    geometry.setIndex(new THREE.BufferAttribute(payload.indices, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.35,
      metalness: 0.1,
      transparent: true,
      opacity: 0.9,
    });

    const mesh = new THREE.Mesh(geometry, material);
    sceneRef.current.add(mesh);
    meshRef.current = mesh;

    const bounds = new THREE.Box3().setFromBufferAttribute(
      geometry.getAttribute("position") as THREE.BufferAttribute
    );
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    mesh.position.sub(center);
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0, maxDim * 1.6 + 10);
      cameraRef.current.lookAt(0, 0, 0);
    }
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setError(null);
    const reader = new FileReader();
    reader.onload = () => {
      setFchkText(String(reader.result || ""));
    };
    reader.readAsText(file);
  };

  const handleRun = () => {
    if (!fchkText || !workerRef.current) return;
    setBusy(true);
    setError(null);
    setStats(null);
    workerRef.current.postMessage({
      type: "compute",
      payload: {
        fchkText,
        gridSize,
        padding,
        aValue,
        isoValue,
        colorRange: { min: colorMin, max: colorMax },
      },
    });
  };

  const handleExport = () => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const dataUrl = renderer.domElement.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = "iri-isosurface.png";
    link.click();
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Interaction Region Indicator</p>
          <h1>Browser IRI Analysis</h1>
          <p className="subtitle">
            Upload a Gaussian <code>.fchk</code> file to compute IRI and render
            a 3D isosurface colored by sign(λ2)ρ.
          </p>
        </div>
        <div className="hero-badge">Closed-shell only</div>
      </header>

      <main className="content">
        <section className="panel">
          <div className="panel-block">
            <label className="label">FCHK file</label>
            <input type="file" accept=".fchk" onChange={onFileChange} />
            <p className="meta">{fileName ?? "No file selected"}</p>
          </div>

          <div className="panel-block">
            <label className="label">Grid preset</label>
            <select
              value={gridSize}
              onChange={(event) => setGridSize(Number(event.target.value))}
            >
              {GRID_PRESETS.map((preset) => (
                <option key={preset.value} value={preset.value}>
                  {preset.label}
                </option>
              ))}
            </select>
            {heavyRun ? (
              <p className="warning">
                Heavy run enabled. Estimated memory ≈ {memoryEstimateMb} MB.
              </p>
            ) : (
              <p className="meta">Estimated memory ≈ {memoryEstimateMb} MB.</p>
            )}
          </div>

          <div className="panel-block two-col">
            <label className="label">Padding (Bohr)</label>
            <input
              type="number"
              step="0.5"
              value={padding}
              onChange={(event) => setPadding(Number(event.target.value))}
            />
            <label className="label">a (IRI)</label>
            <input
              type="number"
              step="0.05"
              value={aValue}
              onChange={(event) => setAValue(Number(event.target.value))}
            />
          </div>

          <div className="panel-block two-col">
            <label className="label">Isovalue</label>
            <input
              type="number"
              step="0.05"
              value={isoValue}
              onChange={(event) => setIsoValue(Number(event.target.value))}
            />
            <label className="label">Color min</label>
            <input
              type="number"
              step="0.01"
              value={colorMin}
              onChange={(event) => setColorMin(Number(event.target.value))}
            />
          </div>

          <div className="panel-block two-col">
            <label className="label">Color max</label>
            <input
              type="number"
              step="0.01"
              value={colorMax}
              onChange={(event) => setColorMax(Number(event.target.value))}
            />
            <div className="color-legend">
              <div className="legend-bar" />
              <div className="legend-labels">
                <span>{colorMin.toFixed(2)}</span>
                <span>0</span>
                <span>{colorMax.toFixed(2)}</span>
              </div>
            </div>
          </div>

          <div className="panel-block actions">
            <button className="primary" onClick={handleRun} disabled={!fchkText || busy}>
              {busy ? "Computing..." : "Compute IRI"}
            </button>
            <button onClick={handleExport} disabled={!meshRef.current}>
              Export PNG
            </button>
          </div>

          {progress && (
            <div className="panel-block">
              <div className="progress">
                <div className="progress-label">
                  {progress.stage} · {progress.percent}%
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${progress.percent}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="panel-block error">
              {error}
            </div>
          )}

          {stats && (
            <div className="panel-block stats">
              <div>
                <span>Basis functions</span>
                <strong>{stats.basisCount}</strong>
              </div>
              <div>
                <span>Occupied orbitals</span>
                <strong>{stats.occOrbitals}</strong>
              </div>
              <div>
                <span>Grid size</span>
                <strong>{stats.gridSize}³</strong>
              </div>
            </div>
          )}

          <div className="panel-block note">
            This implementation follows Lu & Chen (2021): IRI = |∇ρ| / ρ^a,
            with sign(λ2)ρ mapped to color. Defaults: a = 1.1, isovalue = 1.0.
          </div>
        </section>

        <section className="viewer">
          <div className="viewer-header">
            <div>
              <h2>Isosurface View</h2>
              <p>Rotate with mouse, scroll to zoom. Colors show interaction sign.</p>
            </div>
          </div>
          <div className="viewer-canvas" ref={containerRef} />
        </section>
      </main>
    </div>
  );
}

export default App;
