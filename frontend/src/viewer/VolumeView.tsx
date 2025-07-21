import React from "react";
import VolumeRenderer from "./vtkjs/VolumeRenderer"; // wrapper around vtk.js API  :contentReference[oaicite:6]{index=6}
export default function VolumeView({ niftiUrl }: {niftiUrl:string}) {
  return <VolumeRenderer url={niftiUrl} />;
}