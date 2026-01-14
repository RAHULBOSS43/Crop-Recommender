export async function fetchStates() {
  const res = await fetch("http://127.0.0.1:5000/states");
  return res.json();
}

export async function predict(formData) {
  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData,
  });
  return res.json();
}
