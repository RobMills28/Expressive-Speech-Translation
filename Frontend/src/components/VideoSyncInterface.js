import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { AlertCircle, LucideVideo as Video, AudioWaveform, Clock, Settings } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";

const VideoSyncInterface = () => {
  const [video, setVideo] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [syncSettings, setSyncSettings] = useState({
    lipSyncAccuracy: 75,
    speedAdjustment: 100,
    emotionalAlignment: 80
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const videoRef = useRef(null);
  const audioRef = useRef(null);

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideo(URL.createObjectURL(file));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-semibold">Video Synchronization</h2>
          <p className="text-gray-500">Align your translated audio with video</p>
        </div>
        <Button variant="outline" onClick={() => setShowAdvanced(!showAdvanced)}>
          <Settings className="w-4 h-4 mr-2" />
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="w-5 h-5" />
              Video Preview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              {video ? (
                <video
                  ref={videoRef}
                  src={video}
                  className="w-full h-full object-cover"
                  controls
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Button 
                    onClick={() => document.getElementById('video-upload').click()}
                    variant="ghost"
                  >
                    <Video className="w-6 h-6 mr-2" />
                    Upload Video
                  </Button>
                  <input
                    id="video-upload"
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleVideoUpload}
                  />
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Waveform */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AudioWaveform className="w-5 h-5" />
              Audio Waveform
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
              {audioUrl ? (
                <div className="w-full h-full p-4">
                  {/* Placeholder for waveform visualization */}
                  <div className="w-full h-full bg-gradient-to-r from-purple-200 to-purple-100 rounded-lg" />
                </div>
              ) : (
                <div className="text-center">
                  <p className="text-gray-500">Translated audio will appear here</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Synchronization Controls */}
      {showAdvanced && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Synchronization Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">
                  Lip Sync Accuracy
                </label>
                <Slider
                  value={[syncSettings.lipSyncAccuracy]}
                  onValueChange={([value]) => 
                    setSyncSettings(prev => ({ ...prev, lipSyncAccuracy: value }))
                  }
                  max={100}
                  step={1}
                  className="my-2"
                />
                <p className="text-sm text-gray-500">
                  Higher accuracy may result in longer processing time
                </p>
              </div>

              <div>
                <label className="text-sm font-medium">
                  Speed Adjustment
                </label>
                <Slider
                  value={[syncSettings.speedAdjustment]}
                  onValueChange={([value]) => 
                    setSyncSettings(prev => ({ ...prev, speedAdjustment: value }))
                  }
                  max={200}
                  min={50}
                  step={1}
                  className="my-2"
                />
                <p className="text-sm text-gray-500">
                  Adjust video speed to match audio timing
                </p>
              </div>

              <div>
                <label className="text-sm font-medium">
                  Emotional Alignment
                </label>
                <Slider
                  value={[syncSettings.emotionalAlignment]}
                  onValueChange={([value]) => 
                    setSyncSettings(prev => ({ ...prev, emotionalAlignment: value }))
                  }
                  max={100}
                  step={1}
                  className="my-2"
                />
                <p className="text-sm text-gray-500">
                  Match facial expressions with speech emotion
                </p>
              </div>
            </div>

            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                More aggressive synchronization settings may affect video quality
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}

      <div className="flex justify-end gap-4">
        <Button variant="outline">
          Reset
        </Button>
        <Button 
          className="bg-purple-600 hover:bg-purple-700"
          disabled={!video || !audioUrl || isProcessing}
        >
          {isProcessing ? (
            <>
              <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
              Processing...
            </>
          ) : (
            'Synchronize Video'
          )}
        </Button>
      </div>
    </div>
  );
};

export default VideoSyncInterface;