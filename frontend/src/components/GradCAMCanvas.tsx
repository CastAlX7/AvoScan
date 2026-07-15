"use client";

import { useEffect, useRef } from "react";

interface Props {
  imageSrc: string;
  heatmap: number[][];
}

function jetColormap(value: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  let r = 0, g = 0, b = 0;

  if (v < 0.25) {
    r = 0;
    g = Math.round(255 * (v / 0.25));
    b = 255;
  } else if (v < 0.5) {
    r = 0;
    g = 255;
    b = Math.round(255 * (1 - (v - 0.25) / 0.25));
  } else if (v < 0.75) {
    r = Math.round(255 * ((v - 0.5) / 0.25));
    g = 255;
    b = 0;
  } else {
    r = 255;
    g = Math.round(255 * (1 - (v - 0.75) / 0.25));
    b = 0;
  }

  return [r, g, b];
}

export default function GradCAMCanvas({ imageSrc, heatmap }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !heatmap.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = img.width;
      const h = img.height;
      canvas.width = w;
      canvas.height = h;

      ctx.drawImage(img, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const pixels = imageData.data;

      const heatH = heatmap.length;
      const heatW = heatmap[0].length;
      const alpha = 0.4;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const hx = Math.floor((x / w) * heatW);
          const hy = Math.floor((y / h) * heatH);
          const val = heatmap[hy]?.[hx] ?? 0;

          if (val > 0.01) {
            const [hr, hg, hb] = jetColormap(val);
            const idx = (y * w + x) * 4;
            pixels[idx] = Math.round(pixels[idx] * (1 - alpha) + hr * alpha);
            pixels[idx + 1] = Math.round(pixels[idx + 1] * (1 - alpha) + hg * alpha);
            pixels[idx + 2] = Math.round(pixels[idx + 2] * (1 - alpha) + hb * alpha);
          }
        }
      }

      ctx.putImageData(imageData, 0, 0);
    };
    img.src = imageSrc;
  }, [imageSrc, heatmap]);

  return <canvas ref={canvasRef} className="gradcam-canvas" />;
}
