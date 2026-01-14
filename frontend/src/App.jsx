// import { useEffect, useState } from "react";

// export default function App() {
//   const [states, setStates] = useState([]);
//   const [result, setResult] = useState(null);
//   const [form, setForm] = useState({
//     image_url: "",
//     n: "", p: "", k: "",
//     temperature: "", humidity: "",
//     ph: "", rainfall: "",
//     state: ""
//   });

//   useEffect(() => {
//     fetch("http://127.0.0.1:5000/states")
//       .then(res => res.json())
//       .then(setStates);
//   }, []);

//   const submit = async () => {
//     const res = await fetch("http://127.0.0.1:5000/predict", {
//       method: "POST",
//       headers: {"Content-Type": "application/json"},
//       body: JSON.stringify(form)
//     });
//     const data = await res.json();
//     setResult(data);
//   };

//   return (
//     <div className="min-h-screen bg-green-50 p-10">
//       <h1 className="text-3xl font-bold mb-6 text-center">
//         🌱 Soil → Crop Recommendation
//       </h1>

//       <div className="grid grid-cols-2 gap-4 max-w-3xl mx-auto">
//         <input placeholder="Image URL"
//           className="p-2 border"
//           onChange={e=>setForm({...form,image_url:e.target.value})} />

//         <select className="p-2 border"
//           onChange={e=>setForm({...form,state:e.target.value})}>
//           <option>Select State</option>
//           {states.map(s=>(
//             <option key={s}>{s}</option>
//           ))}
//         </select>

//         {["n","p","k","temperature","humidity","ph","rainfall"].map(f=>(
//           <input key={f}
//             placeholder={f.toUpperCase()}
//             className="p-2 border"
//             onChange={e=>setForm({...form,[f]:e.target.value})} />
//         ))}
//       </div>

//       <div className="text-center mt-6">
//         <button
//           onClick={submit}
//           className="bg-green-600 text-white px-6 py-2 rounded">
//           Predict
//         </button>
//       </div>

//       {result && (
//         <div className="mt-8 text-center">
//           <h2 className="text-xl font-bold">
//             Soil: {result.soil}
//           </h2>
//           {result.recommendations.map((r,i)=>(
//             <p key={i}>
//               🌾 {r.crop} — {r.confidence}%
//             </p>
//           ))}
//         </div>
//       )}
//     </div>
//   );
// }

import { useEffect, useState } from "react";

export default function App() {
  const [states, setStates] = useState([]);
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);

  const [form, setForm] = useState({
    n: "", p: "", k: "", temp: "",
    humidity: "", ph: "", rainfall: "", state: ""
  });

  useEffect(() => {
    fetch("http://127.0.0.1:5000/states")
      .then(res => res.json())
      .then(setStates);
  }, []);

  const handlePredict = async () => {
    const data = new FormData();
    Object.entries(form).forEach(([k,v]) => data.append(k,v));
    if (file) data.append("image", file);
    if (url) data.append("image_url", url);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: data
    });

    setResult(await res.json());
  };

  return (
    <div className="min-h-screen bg-green-50 flex justify-center items-center">
      <div className="bg-white p-6 rounded-xl w-[420px] shadow">

        <h1 className="text-xl font-bold mb-4">🌱 Soil → Crop Prediction</h1>

        <input
          className="input"
          placeholder="Paste Image URL"
          onChange={e => setUrl(e.target.value)}
        />

        <input
          type="file"
          className="input mt-2"
          onChange={e => setFile(e.target.files[0])}
        />

        {["n","p","k","temp","humidity","ph","rainfall"].map(k => (
          <input
            key={k}
            placeholder={k.toUpperCase()}
            className="input mt-2"
            onChange={e => setForm({...form,[k]:e.target.value})}
          />
        ))}

        <select
          className="input mt-2"
          onChange={e => setForm({...form,state:e.target.value})}
        >
          <option>Select State</option>
          {states.map(s => <option key={s}>{s}</option>)}
        </select>

        <button
          onClick={handlePredict}
          className="bg-green-600 text-white w-full mt-4 py-2 rounded"
        >
          Predict
        </button>

        {result && (
          <div className="mt-4">
            <p><b>Soil:</b> {result.soil}</p>
            {result.crops.map((c,i) => (
              <p key={i}>{c.name} — {c.percent}%</p>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
