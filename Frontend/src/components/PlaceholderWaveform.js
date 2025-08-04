// src/components/PlaceholderWaveform.js
import React from 'react';

const PlaceholderWaveform = () => {
  // This creates a series of random-height bars to simulate a static waveform
  const bars = Array.from({ length: 50 }, () => Math.random());

  return (
    <div className="w-full h-[100px] flex items-center justify-center bg-gray-50 rounded-lg p-4">
      <div className="flex items-end h-full w-full space-x-1">
        {bars.map((height, i) => (
          <div
            key={i}
            className="w-full bg-fuchsia-300 rounded"
            // This style creates the visual effect of a sound wave
            style={{ height: `${5 + height * 70}%` }}
          ></div>
        ))}
      </div>
    </div>
  );
};

export { PlaceholderWaveform };