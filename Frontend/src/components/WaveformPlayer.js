// src/components/WaveformPlayer.js
import React, { useRef, useEffect, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, Rewind, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { Button } from './ui/button';

const WaveformPlayer = ({ url }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [isMuted, setIsMuted] = useState(false);
  const [duration, setDuration] = useState('0:00');
  const [currentTime, setCurrentTime] = useState('0:00');
  const [isLoading, setIsLoading] = useState(true);

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    if (waveformRef.current && url) {
      setIsLoading(true);
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
      
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#a855f7', // Fuchsia-500
        progressColor: '#7e22ce', // Fuchsia-700
        cursorColor: '#9333ea', // Fuchsia-600
        barWidth: 3,
        barRadius: 3,
        responsive: true,
        height: 100,
        normalize: true,
        url: url,
      });

      wavesurfer.current.on('ready', () => {
        setDuration(formatTime(wavesurfer.current.getDuration()));
        setIsLoading(false);
      });
      wavesurfer.current.on('audioprocess', () => {
        setCurrentTime(formatTime(wavesurfer.current.getCurrentTime()));
      });
      wavesurfer.current.on('play', () => setIsPlaying(true));
      wavesurfer.current.on('pause', () => setIsPlaying(false));
      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
        wavesurfer.current.seekTo(0);
      });

      return () => wavesurfer.current.destroy();
    }
  }, [url]);

  const handlePlayPause = () => wavesurfer.current?.playPause();
  const handleRewind = () => wavesurfer.current?.seekTo(0);
  
  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    wavesurfer.current?.setVolume(newVolume);
    if (newVolume > 0) setIsMuted(false);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    wavesurfer.current?.setVolume(isMuted ? volume : 0);
  };

  return (
    <div className="bg-gray-50 rounded-lg p-4 w-full">
      <div className="relative min-h-[100px] flex items-center justify-center" ref={waveformRef}>
        {isLoading && <Loader2 className="w-8 h-8 animate-spin text-fuchsia-600" />}
      </div>
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center space-x-2">
          <Button variant="ghost" size="icon" onClick={handleRewind} disabled={isLoading}><Rewind className="h-5 w-5" /></Button>
          <Button variant="ghost" size="icon" onClick={handlePlayPause} disabled={isLoading}>
            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </Button>
        </div>
        <div className="text-sm font-mono text-gray-500">{currentTime} / {duration}</div>
        <div className="flex items-center space-x-2 w-28">
          <Button variant="ghost" size="icon" onClick={toggleMute} disabled={isLoading}>
            {isMuted || volume === 0 ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
          </Button>
          <input
            type="range" min="0" max="1" step="0.05"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            disabled={isLoading}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-fuchsia-600"
          />
        </div>
      </div>
    </div>
  );
};

export default WaveformPlayer;