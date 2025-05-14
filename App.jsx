import React, { useState } from "react";

function App() {
  const [status, setStatus] = useState("");
  const [imageName, setImageName] = useState("");

  const handleCapture = async () => {
    setStatus("Capturing...");
    setImageName("");
    try {
      const response = await fetch("http://192.168.73.206:5000/capture");
      const data = await response.json();

      if (data.status === "success") {
        setStatus("Image captured & sent successfully!");
        setImageName(data.filename);
      } else {
        setStatus(`Error: ${data.message}`);
      }
    } catch (error) {
      setStatus("Connection failed!");
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h2>Raspberry Pi Image Capture</h2>
      <button
        style={{ padding: "10px 20px", fontSize: "16px" }}
        onClick={handleCapture}
      >
        Capture Image
      </button>
      <p>{status}</p>
      {imageName && (
        <div>
          <p>Captured Image: {imageName}</p>
        </div>
      )}
    </div>
  );
}

export default App;
