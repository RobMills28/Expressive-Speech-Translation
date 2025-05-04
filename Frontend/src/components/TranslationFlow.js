import React, { useState, useRef } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, Upload, Loader2, Play, Pause, Film, Mic, AudioWaveform } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";

// Language options
const LANGUAGES = {
  'fra': { name: 'French', flag: '🇫🇷' },
  'spa': { name: 'Spanish', flag: '🇪🇸' },
  'deu': { name: 'German', flag: '🇩🇪' },
  'ita': { name: 'Italian', flag: '🇮🇹' },
  'por': { name: 'Portuguese', flag: '🇵🇹' },
  'rus': { name: 'Russian', flag: '🇷🇺' },
  'jpn': { name: 'Japanese', flag: '🇯🇵' },
  'cmn': { name: 'Chinese (Simplified)', flag: '🇨🇳' },
  'ukr': { name: 'Ukrainian', flag: '🇺🇦' }
};

const ContentTranslator = () => {
  // State for selection screen vs main interface
  const [currentScreen, setCurrentScreen] = useState('selection'); // 'selection', 'translator'
  
  // States for managing the flow
  const [contentType, setContentType] = useState(null); // 'audio', 'video', or 'both'
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processPhase, setProcessPhase] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Refs for media elements
  const mediaRef = useRef(null);
  const resultRef = useRef(null);
  const audioRef = useRef(null);
  
  // Select content type and proceed to main interface
  const handleContentTypeSelect = (type) => {
    setContentType(type);
    setCurrentScreen('translator');
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    // Validate file type
    const isAudioType = uploadedFile.type.startsWith('audio/');
    const isVideoType = uploadedFile.type.startsWith('video/');
    
    if (contentType === 'audio' && !isAudioType) {
      setError('Please upload a valid audio file');
      return;
    }
    
    if (contentType === 'video' && !isVideoType) {
      setError('Please upload a valid video file');
      return;
    }
    
    if (contentType === 'both' && !isVideoType) {
      setError('Please upload a valid video file with audio');
      return;
    }

    // Validate file size (50MB limit)
    if (uploadedFile.size > 50 * 1024 * 1024) {
      setError('File size should be less than 50MB');
      return;
    }

    setFile(uploadedFile);
    setFileUrl(URL.createObjectURL(uploadedFile));
    setError('');
    setResult(null);
  };

  // Toggle audio playback
  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Process the translation
  const handleTranslate = async () => {
    if (!file) {
      setError('Please upload a file first');
      return;
    }

    if (!targetLanguage) {
      setError('Please select a target language');
      return;
    }

    try {
      setIsProcessing(true);
      setProgress(0);
      setProcessPhase('Preparing content for processing...');
      setError('');

      const formData = new FormData();
      formData.append(contentType === 'audio' ? 'audio' : 'video', file);
      formData.append('target_language', targetLanguage);

      const endpoint = contentType === 'audio' ? 
        'http://localhost:5001/process-audio' : 
        'http://localhost:5001/process-video';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to process ${contentType}`);
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
                const mimeType = contentType === 'audio' ? 'audio/mp3' : 'video/mp4';
                const blob = new Blob(
                  [Uint8Array.from(atob(data.result), c => c.charCodeAt(0))],
                  { type: mimeType }
                );
                const resultUrl = URL.createObjectURL(blob);
                setResult(resultUrl);
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

  // Reset everything to start over
  const handleReset = () => {
    // Clean up URLs
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    if (result) URL.revokeObjectURL(result);
    
    // Reset state
    setCurrentScreen('selection');
    setContentType(null);
    setFile(null);
    setFileUrl(null);
    setTargetLanguage(null);
    setResult(null);
    setError('');
    setProgress(0);
    setProcessPhase('');
    setIsPlaying(false);
  };

  // Render content selection screen
  const renderSelectionScreen = () => (
    <div className="bg-white rounded-lg shadow-sm p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6 text-center">Choose Content Type</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div 
          className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
          onClick={() => handleContentTypeSelect('audio')}
        >
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <AudioWaveform className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Audio Translation</h3>
          <p className="text-sm text-gray-500 text-center">
            Translate podcasts and audio content
          </p>
        </div>
        
        <div 
          className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
          onClick={() => handleContentTypeSelect('video')}
        >
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <Film className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Video Translation</h3>
          <p className="text-sm text-gray-500 text-center">
            Translate videos with synchronized subtitles
          </p>
        </div>
        
        <div 
          className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
          onClick={() => handleContentTypeSelect('both')}
        >
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <div className="relative">
              <Film className="h-8 w-8 text-fuchsia-600" />
              <AudioWaveform className="h-4 w-4 text-fuchsia-600 absolute -bottom-1 -right-1" />
            </div>
          </div>
          <h3 className="font-medium mb-2">Audio + Video</h3>
          <p className="text-sm text-gray-500 text-center">
            Translate both audio and visual elements
          </p>
        </div>
      </div>
    </div>
  );

  // Render audio interface
  const renderAudioInterface = () => (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">Audio Translation</h1>
          <p className="text-gray-500">Translate speech in your audio files</p>
        </div>
        <Button 
          variant="outline"
          onClick={handleReset}
        >
          Change Content Type
        </Button>
      </div>
      
      {/* Error display */}
      {error && (
        <div className="px-6 pt-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}
      
      {/* Main content area - Side by side audio waveforms */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        {/* Left side - Original audio */}
        <div>
          <h3 className="font-medium text-lg mb-3">Original Audio</h3>
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden mb-3 flex flex-col items-center justify-center">
            {fileUrl ? (
              <>
                <audio 
                  ref={audioRef}
                  src={fileUrl} 
                  className="hidden"
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
                <div className="w-full h-full flex flex-col items-center justify-center p-6">
                  <div className="w-full flex-1 flex items-center justify-center">
                    <div className="w-full h-24 bg-fuchsia-100 rounded-lg relative overflow-hidden">
                      {/* Waveform visualization - stylized version */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="flex items-end h-16 gap-1">
                          {Array.from({ length: 40 }).map((_, i) => {
                            const height = Math.sin(i * 0.5) * 0.5 + 0.5;
                            return (
                              <div 
                                key={i}
                                className="w-1 bg-fuchsia-500"
                                style={{ height: `${height * 100}%` }}
                              />
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4">
                    <Button
                      variant="outline"
                      size="icon"
                      className="rounded-full p-3 bg-white shadow"
                      onClick={handlePlayPause}
                    >
                      {isPlaying ? (
                        <Pause className="h-5 w-5 text-fuchsia-600" />
                      ) : (
                        <Play className="h-5 w-5 text-fuchsia-600" />
                      )}
                    </Button>
                  </div>
                </div>
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <label className="cursor-pointer">
                  <div className="flex flex-col items-center gap-2">
                    <Upload className="w-10 h-10 text-fuchsia-600" />
                    <span className="text-gray-600">Upload Audio</span>
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept="audio/*"
                    onChange={handleFileUpload}
                  />
                </label>
              </div>
            )}
          </div>
          {fileUrl && !result && (
            <div className="flex items-center gap-3">
              <Select 
                value={targetLanguage} 
                onValueChange={setTargetLanguage}
                disabled={isProcessing}
              >
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
                    <SelectItem key={code} value={code}>
                      <span className="flex items-center gap-2">
                        <span>{flag}</span>
                        <span>{name}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                disabled={!file || !targetLanguage || isProcessing}
                onClick={handleTranslate}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Translate'
                )}
              </Button>
            </div>
          )}
        </div>
        
        {/* Right side - Translated audio */}
        <div>
          <h3 className="font-medium text-lg mb-3">
            {targetLanguage && LANGUAGES[targetLanguage] 
              ? `${LANGUAGES[targetLanguage].name} Translation` 
              : 'Translated Audio'}
          </h3>
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
            {result ? (
              <div className="w-full h-full flex flex-col items-center justify-center p-6">
                <audio 
                  ref={resultRef}
                  src={result} 
                  className="hidden"
                  controls
                />
                <div className="w-full flex-1 flex items-center justify-center">
                  <div className="w-full h-24 bg-fuchsia-100 rounded-lg relative overflow-hidden">
                    {/* Waveform visualization for translated audio */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="flex items-end h-16 gap-1">
                        {Array.from({ length: 40 }).map((_, i) => {
                          const height = Math.cos(i * 0.3) * 0.5 + 0.5;
                          return (
                            <div 
                              key={i}
                              className="w-1 bg-fuchsia-500"
                              style={{ height: `${height * 100}%` }}
                            />
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4">
                  <Button
                    variant="outline"
                    size="icon"
                    className="rounded-full p-3 bg-white shadow"
                    onClick={() => {
                      if (resultRef.current) {
                        if (resultRef.current.paused) {
                          resultRef.current.play();
                        } else {
                          resultRef.current.pause();
                        }
                      }
                    }}
                  >
                    <Play className="h-5 w-5 text-fuchsia-600" />
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                {isProcessing ? (
                  <div className="flex flex-col items-center">
                    <Loader2 className="w-8 h-8 animate-spin text-fuchsia-600 mb-2" />
                    <div className="text-center">
                      <p className="mb-2">Processing your audio...</p>
                      <p className="text-sm text-gray-400">{processPhase}</p>
                    </div>
                  </div>
                ) : (
                  "Your translated audio will appear here"
                )}
              </div>
            )}
          </div>
          {isProcessing && (
            <div className="mt-3">
              <Progress value={progress} className="[&>div]:bg-fuchsia-600" />
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Render video interface
  const renderVideoInterface = () => (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">
            {contentType === 'both' ? 'Audio + Video Translation' : 'Video Translation'}
          </h1>
          <p className="text-gray-500">
            {contentType === 'both' 
              ? 'Translate speech and synchronize subtitles in your videos' 
              : 'Translate and synchronize speech in your videos'}
          </p>
        </div>
        <Button 
          variant="outline"
          onClick={handleReset}
        >
          Change Content Type
        </Button>
      </div>
      
      {/* Error display */}
      {error && (
        <div className="px-6 pt-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}
      
      {/* Main content area - Side by side videos */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        {/* Left side - Original video */}
        <div>
          <h3 className="font-medium text-lg mb-3">Video Preview</h3>
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden mb-3">
            {fileUrl ? (
              <video
                ref={mediaRef}
                src={fileUrl}
                className="w-full h-full object-contain"
                controls
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <label className="cursor-pointer">
                  <div className="flex flex-col items-center gap-2">
                    <Upload className="w-10 h-10 text-fuchsia-600" />
                    <span className="text-gray-600">Upload Video</span>
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept="video/*"
                    onChange={handleFileUpload}
                  />
                </label>
              </div>
            )}
          </div>
          {fileUrl && !result && (
            <div className="flex items-center gap-3">
              <Select 
                value={targetLanguage} 
                onValueChange={setTargetLanguage}
                disabled={isProcessing}
              >
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
                    <SelectItem key={code} value={code}>
                      <span className="flex items-center gap-2">
                        <span>{flag}</span>
                        <span>{name}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                disabled={!file || !targetLanguage || isProcessing}
                onClick={handleTranslate}
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
          )}
        </div>
        
        {/* Right side - Translated video */}
        <div>
          <h3 className="font-medium text-lg mb-3">
            {targetLanguage && LANGUAGES[targetLanguage] 
              ? `${LANGUAGES[targetLanguage].name} Translation` 
              : 'Translated Result'}
          </h3>
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
            {result ? (
              <video
                ref={resultRef}
                src={result}
                className="w-full h-full object-contain"
                controls
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                {isProcessing ? (
                  <div className="flex flex-col items-center">
                    <Loader2 className="w-8 h-8 animate-spin text-fuchsia-600 mb-2" />
                    <div className="text-center">
                      <p className="mb-2">Processing your video...</p>
                      <p className="text-sm text-gray-400">{processPhase}</p>
                    </div>
                  </div>
                ) : (
                  "Your translated video will appear here"
                )}
              </div>
            )}
          </div>
          {isProcessing && (
            <div className="mt-3">
              <Progress value={progress} className="[&>div]:bg-fuchsia-600" />
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Render the current screen
  return (
    <div className="mx-auto px-4 py-6 max-w-7xl">
      {currentScreen === 'selection' ? (
        renderSelectionScreen()
      ) : (
        contentType === 'audio' ? renderAudioInterface() : renderVideoInterface()
      )}
    </div>
  );
};

export default ContentTranslator;