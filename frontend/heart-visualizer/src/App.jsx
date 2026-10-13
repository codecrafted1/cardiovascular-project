
import { useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Center,
  useGLTF,
  Html,
  Bounds,
} from "@react-three/drei";
import "./App.css";

const vessels = {
  LAD: {
    color: "#ef4444",
    probability: 81,
    description: "Left Anterior Descending artery",
    position: [-1.15, -0.45, 0.2],
  },
  LCX: {
    color: "#22c55e",
    probability: 35,
    description: "Left Circumflex artery",
    position: [1.05, 0.25, 0.2],
  },
  RCA: {
    color: "#facc15",
    probability: 64,
    description: "Right Coronary Artery",
    position: [1.05, -0.65, 0.2],
  },
};

// Temporary demonstration data.
// Replace with real SHAP output from your teammate.
const demoFactors = [
  { name: "Age", contribution: 22 },
  { name: "Cholesterol", contribution: 18 },
  { name: "ST Elevation", contribution: 16 },
  { name: "Blood Pressure", contribution: 12 },
  { name: "Pulse Rate", contribution: 8 },
];

function HeartModel() {
  const { scene } = useGLTF("/models/heart_clear.glb");

  return (
    <Bounds fit clip observe margin={1.35}>
      <Center>
        <primitive object={scene} />
      </Center>
    </Bounds>
  );
}

function VesselLabel({ name, onSelect, selected }) {
  const vessel = vessels[name];

  return (
    <Html
      position={vessel.position}
      center
      distanceFactor={7}
      style={{ pointerEvents: "auto" }}
    >
      <button
        className={`heart-label ${selected ? "label-active" : ""}`}
        style={{ "--vessel-color": vessel.color }}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(name);
        }}
      >
        <span>{name}</span>
        <strong>{vessel.probability}%</strong>
      </button>
    </Html>
  );
}

function App() {
  const [selected, setSelected] = useState("LAD");
  const controlsRef = useRef();

  const vessel = vessels[selected];

  function resetView() {
    controlsRef.current?.reset();
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">♥</div>
          <div>
            <h1>CardioVision AI</h1>
            <p>3D Coronary Artery Risk Visualization</p>
          </div>
        </div>

        <span className="status">DEMO MODE</span>
      </header>

      <main className="dashboard">
        <section className="heart-panel">
          <div className="panel-heading">
            <div>
              <h2>Interactive 3D Heart</h2>
              <p>Drag to rotate · Scroll to zoom</p>
            </div>
          </div>

          <div className="canvas-container">
            <Canvas
              camera={{ position: [0, 0, 6], fov: 40 }}
              dpr={[1, 1.5]}
              gl={{ antialias: true }}
            >
              <color attach="background" args={["#0f172a"]} />

              <ambientLight intensity={1.8} />
              <directionalLight
                position={[5, 6, 5]}
                intensity={2.2}
              />
              <directionalLight
                position={[-5, 2, -5]}
                intensity={1.2}
              />

              <HeartModel />

              {Object.keys(vessels).map((name) => (
                <VesselLabel
                  key={name}
                  name={name}
                  selected={selected === name}
                  onSelect={setSelected}
                />
              ))}

              <OrbitControls
                ref={controlsRef}
                makeDefault
                enablePan={false}
                enableZoom
                minDistance={2}
                maxDistance={12}
                rotateSpeed={0.8}
              />
            </Canvas>
          </div>

          <div className="viewer-controls">
            <button onClick={resetView}>↻ Reset View</button>
            <span>Mouse drag: Rotate</span>
            <span>Scroll: Zoom</span>
          </div>

          <div className="legend">
            {Object.entries(vessels).map(([name, data]) => (
              <button
                key={name}
                className={
                  selected === name
                    ? "legend-item active"
                    : "legend-item"
                }
                onClick={() => setSelected(name)}
              >
                <span
                  className="dot"
                  style={{ backgroundColor: data.color }}
                />
                {name}
                <strong>{data.probability}%</strong>
              </button>
            ))}
          </div>

          <div className="risk-legend">
            <strong>Probability scale</strong>
            <span><i className="scale-dot green" /> 0–30%</span>
            <span><i className="scale-dot yellow" /> 30–60%</span>
            <span><i className="scale-dot red" /> 60–100%</span>
          </div>
        </section>

        <aside className="details-panel">
          <div className="details-heading">
            <h2>Vessel Details</h2>
            <p>Select a coronary artery</p>
          </div>

          {Object.entries(vessels).map(([name, data]) => (
            <button
              key={name}
              className={
                selected === name
                  ? "vessel-card selected"
                  : "vessel-card"
              }
              onClick={() => setSelected(name)}
            >
              <div className="vessel-title">
                <span
                  className="dot"
                  style={{ backgroundColor: data.color }}
                />
                <strong>{name}</strong>
                <span className="percentage">
                  {data.probability}%
                </span>
              </div>

              <p>{data.description}</p>

              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${data.probability}%`,
                    backgroundColor: data.color,
                  }}
                />
              </div>
            </button>
          ))}

          <div className="selected-details">
            <span
              className="selected-dot"
              style={{ backgroundColor: vessel.color }}
            />
            <h3>Selected: {selected}</h3>
            <p>{vessel.description}</p>
            <div className="selected-probability">
              <span>Demo stenosis probability</span>
              <strong style={{ color: vessel.color }}>
                {vessel.probability}%
              </strong>
            </div>
          </div>

          <section className="factors-panel">
            <h2>Key Contributing Factors</h2>
            <p className="factor-subtitle">
              Example feature contributions for {selected}
            </p>

            {demoFactors.map((factor) => (
              <div className="factor-row" key={factor.name}>
                <div className="factor-heading">
                  <span>{factor.name}</span>
                  <strong>{factor.contribution}%</strong>
                </div>

                <div className="progress-track factor-track">
                  <div
                    className="progress-fill factor-fill"
                    style={{
                      width: `${factor.contribution * 4}%`,
                    }}
                  />
                </div>
              </div>
            ))}

            <p className="factor-note">
              Illustrative values only. Actual feature importance
              must come from the trained model's SHAP explanation.
            </p>

            <button
              className="shap-button"
              onClick={() => {
                window.alert(
                  "SHAP integration will be added when the explanation module is connected."
                );
              }}
            >
              View SHAP Explanation →
            </button>
          </section>

          <p className="disclaimer">
            Educational prototype only. Example probabilities are
            not real patient predictions and must not be used for
            diagnosis.
          </p>
        </aside>
      </main>
    </div>
  );
}

useGLTF.preload("/models/heart_clear.glb");

export default App;


       