import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, Video, AudioWaveform, Loader2 } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";

const SUPPORTED_LANGUAGES = {
  // Most common languages first
  'fra': 'French',
  'spa': 'Spanish',
  'deu': 'German',
  'ita': 'Italian',
  'por': 'Portuguese',
  'rus': 'Russian',
  'jpn': 'Japanese',
  'cmn': 'Chinese (Simplified)',
  'ukr': 'Ukrainian',
  
  // Rest in alphabetical order
  'ben': 'Bengali',
  'cat': 'Catalan',
  'cmn_Hant': 'Chinese (Traditional)',
  'cym': 'Welsh',
  'dan': 'Danish',
  'eng': 'English',
  'est': 'Estonian',
  'fin': 'Finnish',
  'hin': 'Hindi',
  'ind': 'Indonesian',
  'kor': 'Korean',
  'mlt': 'Maltese',
  'nld': 'Dutch',
  'pes': 'Persian',
  'pol': 'Polish',
  'ron': 'Romanian',
  'slk': 'Slovak',
  'swe': 'Swedish',
  'swh': 'Swahili',
  'tel': 'Telugu',
  'tgl': 'Tagalog',
  'tha': 'Thai',
  'tur': 'Turkish',
  'urd': 'Urdu',
  'uzn': 'Uzbek',
  'vie': 'Vietnamese'
};

// Language flags mapping
const LANGUAGE_FLAGS = {
  // Most common languages first
  'fra': '🇫🇷',
  'spa': '🇪🇸',
  'deu': '🇩🇪',
  'ita': '🇮🇹',
  'por': '🇵🇹',
  'rus': '🇷🇺',
  'jpn': '🇯🇵',
  'cmn': '🇨🇳',
  'ukr': '🇺🇦',
  
  // Rest in alphabetical order
  'ben': '🇧🇩',
  'cat': '🏴󠁥󠁳󠁣󠁴󠁿',
  'cmn_Hant': '🇹🇼',
  'cym': '🏴󠁧󠁢󠁷󠁬󠁳󠁿',
  'dan': '🇩🇰',
  'eng': '🇬🇧',
  'est': '🇪🇪',
  'fin': '🇫🇮',
  'hin': '🇮🇳',
  'ind': '🇮🇩',
  'kor': '🇰🇷',
  'mlt': '🇲🇹',
  'nld': '🇳🇱',
  'pes': '🇮🇷',
  'pol': '🇵🇱',
  'ron': '🇷🇴',
  'slk': '🇸🇰',
  'swe': '🇸🇪',
  'swh': '🇹🇿',
  'tel': '🇮🇳',
  'tgl': '🇵🇭',
  'tha': '🇹🇭',
  'tur': '🇹🇷',
  'urd': '🇵🇰',
  'uzn': '🇺🇿',
  'vie': '🇻🇳'
};

const VideoSyncInterface = () => {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('fra');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processPhase, setProcessPhase] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  
  const videoRef = useRef(null);
  const resultRef = useRef(null);

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Please upload a valid video file');
      return;
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      setError('Video file size should be less than 50MB');
      return;
    }

    setVideo(file);
    setVideoUrl(URL.createObjectURL(file));
    setError('');
    setResult(null);
  };

  const handleProcess = async () => {
    try {
      setIsProcessing(true);
      setProgress(0);
      setProcessPhase('Preparing video for processing...');
      setError('');

      const formData = new FormData();
      formData.append('video', video);
      formData.append('target_language', targetLanguage);

      const response = await fetch('http://localhost:5001/process-video', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to process video');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = ''; // Buffer for incomplete messages

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append new data to buffer and split into messages
        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split('\n\n');
        
        // Keep the last item in buffer if it's incomplete
        buffer = messages.pop() || '';

        // Process complete messages
        for (const message of messages) {
          if (message.trim().startsWith('data: ')) {
            try {
              const jsonStr = message.trim().slice(6); // Remove 'data: ' prefix
              const data = JSON.parse(jsonStr);
              
              if (data.error) {
                throw new Error(data.error);
              }
              
              if (data.progress !== undefined) {
                setProgress(data.progress);
                setProcessPhase(data.phase || '');
              }
              
              if (data.result) {
                // Convert base64 to blob and create URL
                const videoBlob = new Blob(
                  [Uint8Array.from(atob(data.result), c => c.charCodeAt(0))],
                  { type: 'video/mp4' }
                );
                const videoUrl = URL.createObjectURL(videoBlob);
                setResult(videoUrl);
              }
            } catch (e) {
              console.error('Error parsing SSE message:', e);
              if (e.message !== 'Unexpected end of JSON input') {
                throw e;
              }
            }
          }
        }
      }

    } catch (err) {
      console.error('Process error:', err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-semibold">Video Translation & Sync</h2>
          <p className="text-gray-500">Translate and synchronize speech in your videos</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Upload/Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="w-5 h-5" />
              Video Preview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              {videoUrl ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
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

        {/* Result Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AudioWaveform className="w-5 h-5" />
              Translated Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              {result ? (
                <video
                  ref={resultRef}
                  src={result}
                  className="w-full h-full object-cover"
                  controls
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-500">
                  Translated video will appear here
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Processing Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Select 
                value={targetLanguage} 
                onValueChange={setTargetLanguage}
                disabled={isProcessing}
              >
                <SelectTrigger className="w-60">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent className="max-h-80 overflow-y-auto">
                  {Object.entries(SUPPORTED_LANGUAGES).map(([code, name]) => (
                    <SelectItem key={code} value={code}>
                      <span>{LANGUAGE_FLAGS[code]}</span> <span className="ml-2">{name}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button
                className="bg-purple-600 hover:bg-purple-700"
                disabled={!video || isProcessing}
                onClick={handleProcess}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Translate & Synchronize'
                )}
              </Button>
            </div>

            {isProcessing && (
              <div className="space-y-2">
                <Progress value={progress} />
                <p className="text-sm text-gray-500">{processPhase}</p>
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default VideoSyncInterface;