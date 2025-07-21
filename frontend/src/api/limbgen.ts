import axios from "axios";
export const upload = (file: File, meta: Blob) =>
  axios.post("/upload", { file, metadata: meta },
             { headers: { "Content-Type": "multipart/form-data" }});