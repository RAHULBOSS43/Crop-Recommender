export default function ImageUpload({ setFormData }) {
  return (
    <input
      type="text"
      placeholder="Paste image URL or upload below"
      className="border p-2 w-full rounded"
      onChange={(e) =>
        setFormData(prev => ({ ...prev, image_url: e.target.value }))
      }
    />
  );
}
