export default function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className="bg-green-100 p-4 rounded">
      <h2>Soil: {result.soil}</h2>
      {result.crops.map((c,i)=>(
        <p key={i}>{i+1}. {c[0]} — {c[1]}%</p>
      ))}
    </div>
  );
}
