import { useEffect, useState } from "react";
import { fetchStates } from "../api";

export default function SoilForm({ formData, setFormData }) {
  const [states, setStates] = useState([]);

  useEffect(() => {
    fetchStates().then(setStates);
  }, []);

  return (
    <select
      value={formData.state}
      onChange={(e) =>
        setFormData({ ...formData, state: e.target.value })
      }
      className="border p-2 rounded w-full"
    >
      <option value="">Select State</option>
      {states.map((state) => (
        <option key={state} value={state}>
          {state}
        </option>
      ))}
    </select>
  );
}
