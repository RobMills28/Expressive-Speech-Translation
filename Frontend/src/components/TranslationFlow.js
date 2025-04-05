import React, { useState, useRef, useCallback } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Upload, Globe, Play, Pause, CheckCircle, AlertCircle, ChevronDown, Check, Film, Mic, Loader2 } from 'lucide-react';
import { useTranslation } from '../hooks/useTranslation';
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import BackendSelector from './BackendSelector';
import TranscriptView from "./ui/TranscriptView";

const TranslationFlow = () => {
  // States from the translation hook
  const {
    audioStatus,
    audioReady,
    file,
    targetLanguage,
    processing: audioProcessing,
    progress: audioProgress,
    error: hookError,
    translatedAudioUrl,
    isPlaying,
    progressText,
    sourceText,
    targetText,
    audioRef,
    handleFileChange: hookHandleFileChange,
    handleLanguageSelect,
    handlePlayPause,
    processAudio,
    cleanup,
    setError: setHookError,
    setAudioStatus,
    setAudioReady,
    setIsPlaying,
    setSourceText,
    setTargetText
  } = useTranslation();

  // Local states for the flow UI
  const [step, setStep] = useState(1);
  const [selectedLanguages, setSelectedLanguages] = useState([]);
  const [languageDropdownOpen, setLanguageDropdownOpen] = useState(false);
  const [translations, setTranslations] = useState({});
  const [mediaType, setMediaType] = useState(null); // 'audio', 'video', or 'both'
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [resultVideo, setResultVideo] = useState(null);
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [processPhase, setProcessPhase] = useState('');
  const [error, setError] = useState('');
  const [selectedBackend, setSelectedBackend] = useState('seamless'); // Default backend
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const resultVideoRef = useRef(null);
  const eventSourceRef = useRef(null);


  // Supported file types
  const AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.m4a'];
  const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.webm', '.avi', '.mkv'];

  const languages = {
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

  const handleMediaTypeSelect = (type) => {
    setMediaType(type);
    setStep(2);
  };

  // Handle backend change
  const handleBackendChange = (backend) => {
    setSelectedBackend(backend);
    console.log(`Translation backend changed to: ${backend}`);
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const validateFile = useCallback((file) => {
    if (!file) return false;
    
    const fileName = file.name.toLowerCase();
    const fileExtension = fileName.substring(fileName.lastIndexOf('.'));
    
    if (mediaType === 'audio') {
      // Audio-only validation
      if (!AUDIO_EXTENSIONS.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${AUDIO_EXTENSIONS.join(', ')}`);
        return false;
      }
    } else if (mediaType === 'video') {
      // Video-only validation
      if (!VIDEO_EXTENSIONS.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${VIDEO_EXTENSIONS.join(', ')}`);
        return false;
      }
    } else {
      // Both audio and video allowed
      if (!AUDIO_EXTENSIONS.includes(fileExtension) && !VIDEO_EXTENSIONS.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${AUDIO_EXTENSIONS.join(', ')} or ${VIDEO_EXTENSIONS.join(', ')}`);
        return false;
      }
    }
    
    return true;
  });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleFileUpload = useCallback((event) => {
    const uploadedFile = event.target.files?.[0];
    setError(''); // Clear any previous error
    
    if (!uploadedFile) return;
    
    // Validate the file
    if (!validateFile(uploadedFile)) {
      return;
    }
    
    const fileName = uploadedFile.name.toLowerCase();
    const fileExtension = fileName.substring(fileName.lastIndexOf('.'));
    const isVideoFile = VIDEO_EXTENSIONS.includes(fileExtension);
    
    // Handle video files
    if (isVideoFile) {
      setVideoFile(uploadedFile);
      setVideoUrl(URL.createObjectURL(uploadedFile));
      
      // If it's video-only, we don't need to call the audio hook
      if (mediaType !== 'both') {
        setStep(3);
        return;
      }
    }
    
    // For audio or both, we still need to call the hook's file handler
    if (mediaType === 'audio' || mediaType === 'both') {
      hookHandleFileChange(event);
    }
    
    setStep(3);
  });

  const toggleLanguage = (code) => {
    if (selectedLanguages.includes(code)) {
      if (selectedLanguages.length > 1) {
        setSelectedLanguages(selectedLanguages.filter(lang => lang !== code));
      }
    } else {
      setSelectedLanguages([...selectedLanguages, code]);
      handleLanguageSelect(code);
    }
    
    // When a language is selected from dropdown, close it
    if (languageDropdownOpen) {
      setLanguageDropdownOpen(false);
    }
  };

  // Clean up event source
  const cleanupEventSource = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const handleVideoTranslation = async (targetLang) => {
    if (!videoFile) {
      setError('No video file selected');
      return false;
    }

    try {
      setProcessing(true);
      setProgress(0);
      setProcessPhase('Preparing video for processing...');
      setError('');
      
      // Clean up previous event source if exists
      cleanupEventSource();

      // Create FormData
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('target_language', targetLang);
      formData.append('backend', selectedBackend); // Add backend selection

      // This exact endpoint worked in VideoSyncInterface
      const response = await fetch('http://localhost:5001/process-video', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to process video');
      }

      // Set up event stream reader, using same format as original
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
                setResultVideo(videoUrl);
              }

              // Extract transcript data if available
              if (data.transcripts) {
                setSourceText(data.transcripts.source || '');
                setTargetText(data.transcripts.target || '');
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
      
      setProcessing(false);
      return true;
    } catch (err) {
      console.error('Process error:', err);
      setError(err.message);
      setProcessing(false);
      return false;
    }
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleContinue = useCallback(async () => {
    if (selectedLanguages.length === 0) {
      setError('Please select at least one language');
      return;
    }
    
    setStep(4);
    setError('');
    
    try {
      // For video or both types, process the video
      if ((mediaType === 'video' || mediaType === 'both') && videoFile) {
        // Use the first selected language for video 
        const targetLang = selectedLanguages[0];
        const success = await handleVideoTranslation(targetLang);
        
        if (!success && mediaType === 'video') {
          return; // Stop if video-only and failed
        }
      }
      
      // If we also have audio to process (audio-only or both)
      if ((mediaType === 'audio' || mediaType === 'both') && file) {
        // Include the selected backend in the processAudio call
        await processAudio(selectedBackend);
        
        // Update translations state
        setTranslations(prev => ({
          ...prev,
          [targetLanguage]: {
            audioUrl: translatedAudioUrl,
            sourceText,
            targetText
          }
        }));
      }
    } catch (e) {
      console.error('Translation error:', e);
      setError(e.message || 'Failed to process translation');
    }
  });

  const handleBackStep = () => {
    if (step > 1) {
      setStep(step - 1);
      
      // Clean up resources when going back
      if (step === 4) {
        // Clean up event source if active
        cleanupEventSource();
        
        // Reset processing state
        setProcessing(false);
        setProgress(0);
      }
    }
  };

  // Clean up on unmount
  React.useEffect(() => {
    return () => {
      cleanupEventSource();
      
      // Clean up any object URLs
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      if (resultVideo) URL.revokeObjectURL(resultVideo);
    };
  }, [videoUrl, resultVideo]);

  // Combine errors from hook and local state
  const combinedError = error || hookError;
  const isProcessing = processing || audioProcessing;

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      {/* Progress Steps - Perfectly Centered */}
      <div className="flex justify-center mb-12">
        <div className="flex items-center justify-between" style={{ width: '280px' }}>
          {[1, 2, 3].map((number) => (
            <div key={number} className="flex items-center">
              <div className={`
                w-10 h-10 rounded-full flex items-center justify-center
                ${step > number ? 'bg-green-500 text-white' : 
                  step === number ? 'bg-fuchsia-600 text-white' : 
                  'bg-gray-200 text-gray-600'}
              `}>
                {step > number ? <CheckCircle className="w-5 h-5" /> : number}
              </div>
              {number < 3 && (
                <div className={`w-16 h-1 mx-1 ${
                  step > number ? 'bg-green-500' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: Choose Content Type */}
      {step === 1 && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-center mb-8">Choose Content Type</h2>
          
          {/* Add Backend Selector above content type options */}
          <div className="mb-8">
            <BackendSelector 
              onBackendChange={handleBackendChange}
              className="max-w-xs mx-auto"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div 
              className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
              onClick={() => handleMediaTypeSelect('audio')}
            >
              <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
                <Mic className="h-5 w-5 text-fuchsia-600" />
              </div>
              <h3 className="font-medium mb-2">Audio Translation</h3>
              <p className="text-sm text-gray-500 text-center">
                Translate podcasts and audio content
              </p>
            </div>
            
            <div 
              className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
              onClick={() => handleMediaTypeSelect('video')}
            >
              <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
                <Film className="h-5 w-5 text-fuchsia-600" />
              </div>
              <h3 className="font-medium mb-2">Video Translation</h3>
              <p className="text-sm text-gray-500 text-center">
                Translate videos with synchronized subtitles
              </p>
            </div>
            
            <div 
              className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
              onClick={() => handleMediaTypeSelect('both')}
            >
              <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
                <Upload className="h-5 w-5 text-fuchsia-600" />
              </div>
              <h3 className="font-medium mb-2">Audio + Video</h3>
              <p className="text-sm text-gray-500 text-center">
                Translate both audio and visual elements
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Upload */}
      {step === 2 && (
        <Card>
          <CardContent className="p-6">
            {combinedError && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{combinedError}</AlertDescription>
              </Alert>
            )}
            
            <div className="text-center">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-12 cursor-pointer hover:border-fuchsia-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-fuchsia-600" />
                <h2 className="text-xl font-semibold mb-2">Upload Your Content</h2>
                <p className="text-gray-600 mb-4">
                  {mediaType === 'audio' ? 'Drag and drop your audio file here' :
                   mediaType === 'video' ? 'Drag and drop your video file here' :
                   'Drag and drop your audio or video file here'}
                </p>
                <Button className="bg-fuchsia-600 hover:bg-fuchsia-700">
                  Select File
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileUpload}
                  accept={
                    mediaType === 'audio' ? 'audio/*' : 
                    mediaType === 'video' ? 'video/*' : 
                    'audio/*,video/*'
                  }
                />
              </div>
              <div className="mt-4 flex justify-start">
                <Button 
                  variant="outline" 
                  onClick={handleBackStep}
                  className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
                >
                  Back
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Language Selection */}
      {step === 3 && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          {combinedError && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{combinedError}</AlertDescription>
            </Alert>
          )}
          
          <h2 className="text-xl font-semibold mb-6">Choose Languages</h2>
          
          {/* Display current backend selection */}
          <div className="mb-6 p-3 bg-fuchsia-50 rounded-md">
            <p className="text-sm text-fuchsia-700">
              Translation Engine: <strong>{selectedBackend === 'seamless' ? 'Seamless' : 'ESPnet (Experimental)'}</strong>
              {selectedBackend === 'espnet' && ' - Limited language support available'}
            </p>
          </div>
          
          {mediaType === 'video' || mediaType === 'both' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Video preview on the left */}
              {videoUrl && (
                <div className="bg-gray-100 rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="w-full h-full object-cover"
                    controls
                  />
                </div>
              )}

              {/* Language selection on the right */}
              <div>
                <div className="relative mb-6">
                  <Button 
                    variant="outline" 
                    className="pl-3 pr-2 py-5 flex items-center justify-between w-full border-gray-300 text-left font-normal"
                    onClick={() => setLanguageDropdownOpen(!languageDropdownOpen)}
                  >
                    <div className="flex items-center gap-2">
                      {selectedLanguages.length === 0 ? (
                        <span>Select languages</span>
                      ) : selectedLanguages.length === 1 ? (
                        <>
                          <span>{languages[selectedLanguages[0]].flag}</span>
                          <span>{languages[selectedLanguages[0]].name}</span>
                        </>
                      ) : (
                        <span>{selectedLanguages.length} languages selected</span>
                      )}
                    </div>
                    <ChevronDown className="h-4 w-4 opacity-50" />
                  </Button>
                  
                  {languageDropdownOpen && (
                    <div className="absolute mt-1 w-full rounded-md bg-gray-800 text-white shadow-lg z-10 max-h-64 overflow-y-auto">
                      {Object.entries(languages).map(([code, { name, flag }]) => (
                        <div 
                          key={code}
                          className="px-4 py-3 flex items-center hover:bg-gray-700 cursor-pointer"
                          onClick={() => toggleLanguage(code)}
                        >
                          {selectedLanguages.includes(code) && (
                            <Check className="w-4 h-4 mr-2 text-fuchsia-400" />
                          )}
                          <span className="ml-2">{flag}</span>
                          <span className="ml-2">{name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            // Original checkbox design for audio only
            <div className="space-y-3 mb-6">
              {Object.entries(languages).map(([code, { name, flag }]) => (
                <label
                  key={code}
                  className={`flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer ${
                    selectedLanguages.includes(code) ? 'border-fuchsia-300 bg-fuchsia-50' : 'border-gray-200'
                  }`}
                >
                  <input
                    type="checkbox"
                    className="w-5 h-5 text-fuchsia-600 border-fuchsia-300 focus:ring-fuchsia-500"
                    checked={selectedLanguages.includes(code)}
                    onChange={() => toggleLanguage(code)}
                  />
                  <Globe className={`w-5 h-5 ml-3 mr-2 ${
                    selectedLanguages.includes(code) ? 'text-fuchsia-500' : 'text-gray-400'
                  }`} />
                  <span>{flag} {name}</span>
                </label>
              ))}
            </div>
          )}
          
          <div className="flex justify-between">
            <Button 
              variant="outline" 
              onClick={handleBackStep}
              className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
            >
              Back
            </Button>
            <Button
              onClick={handleContinue}
              disabled={selectedLanguages.length === 0 || isProcessing}
              className="bg-fuchsia-600 hover:bg-fuchsia-700 disabled:bg-gray-300"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : 'Continue'}
            </Button>
          </div>
        </div>
      )}

      {/* Step 4: Preview Translations */}
      {step === 4 && (
        <Card>
          <CardContent className="p-6">
            {combinedError && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{combinedError}</AlertDescription>
              </Alert>
            )}
            
            <h2 className="text-xl font-semibold mb-6">Preview Translations</h2>
            
            {/* Show which backend was used */}
            <div className="mb-6 p-3 bg-gray-50 rounded-md">
              <p className="text-sm text-gray-600">
                Translation Engine: <strong>{selectedBackend === 'seamless' ? 'Seamless' : 'ESPnet (Experimental)'}</strong>
              </p>
            </div>
            
            {/* For video content */}
            {(mediaType === 'video' || mediaType === 'both') && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Original video */}
                <div>
                  <h3 className="font-medium mb-2">Original</h3>
                  <div className="bg-gray-100 rounded-lg overflow-hidden">
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      className="w-full h-full object-cover"
                      controls
                    />
                  </div>
                </div>
                
                {/* Translated video */}
                <div>
                  <h3 className="font-medium mb-2">
                    {selectedLanguages.length === 1 
                      ? `${languages[selectedLanguages[0]].name} Translation`
                      : 'Translated Result'}
                  </h3>
                  <div className="bg-gray-100 rounded-lg overflow-hidden">
                    {resultVideo ? (
                      <video
                        ref={resultVideoRef}
                        src={resultVideo}
                        className="w-full h-full object-cover"
                        controls
                      />
                    ) : (
                      <div className="flex items-center justify-center h-48 text-gray-500">
                        {isProcessing ? (
                          <div className="flex flex-col items-center">
                            <Loader2 className="w-8 h-8 animate-spin text-fuchsia-600 mb-2" />
                            <p>Processing your video...</p>
                          </div>
                        ) : (
                          "Translated video will appear here"
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            {/* Add Transcript section for video translations */}
            {(mediaType === 'video' || mediaType === 'both') && resultVideo && (
              <div className="mt-6 mb-6">
                <TranscriptView 
                  sourceText={sourceText}
                  targetText={targetText}
                  targetLang={selectedLanguages[0]}
                  sourceLang="English"
                  isOpen={true}
                />
              </div>
            )}
            
            {/* For audio content */}
            {mediaType === 'audio' && (
              <>
                {/* Hidden audio element */}
                <audio 
                  ref={audioRef} 
                  src={translatedAudioUrl}
                  className="hidden"
                  preload="auto"
                  onLoadStart={() => {
                    setAudioReady(false);
                    setAudioStatus('loading');
                  }}
                  onCanPlay={() => {
                    setAudioStatus('ready');
                    setAudioReady(true);
                    setError('');
                  }}
                  onEnded={() => {
                    setIsPlaying(false);
                    setAudioStatus('ready');
                  }}
                  onError={(e) => {
                    console.error('Audio error:', e);
                    setError('Failed to load audio');
                    setAudioStatus('error');
                    setAudioReady(false);
                    setIsPlaying(false);
                  }}
                />
                
                {/* Original Audio */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium">Original</h3>
                    {sourceText && (
                      <p className="text-sm text-gray-600">{sourceText}</p>
                    )}
                  </div>
                  <div className="bg-gray-100 h-12 rounded-lg flex items-center justify-center">
                    {file && <p className="text-sm text-gray-600">{file.name}</p>}
                  </div>
                </div>

                {/* Translated Audio */}
                {selectedLanguages.map(code => {
                  const language = languages[code];
                  const translation = translations[code];
                  
                  return (
                    <div key={code} className="mb-6">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{language?.name}</h3>
                        <Button 
                          size="sm" 
                          variant="outline"
                          className="border-fuchsia-200 text-fuchsia-700 hover:bg-fuchsia-50"
                          onClick={handlePlayPause}
                          disabled={!audioReady || audioStatus === 'error'}
                        >
                          {isPlaying ? 
                            <Pause className="w-4 h-4" /> : 
                            <Play className="w-4 h-4" />
                          }
                        </Button>
                      </div>
                      <div className="bg-gray-100 h-12 rounded-lg flex items-center justify-center">
                        {translation?.targetText && (
                          <p className="text-sm text-gray-600">{translation.targetText}</p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            <div className="flex justify-between mt-6">
              <Button 
                variant="outline"
                className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
                onClick={handleBackStep}
              >
                Back
              </Button>
              <Button 
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                onClick={() => {
                  // Clean up resources
                  cleanup();
                  cleanupEventSource();
                  
                  // Reset all states
                  setStep(1);
                  setVideoFile(null);
                  if (videoUrl) URL.revokeObjectURL(videoUrl);
                  setVideoUrl(null);
                  if (resultVideo) URL.revokeObjectURL(resultVideo);
                  setResultVideo(null);
                  setSelectedLanguages([]);
                  setError('');
                  setProcessing(false);
                  setProgress(0);
                }}
              >
                Start New Translation
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="mt-4">
          <Progress 
            value={mediaType === 'audio' ? audioProgress : progress} 
            className="mb-2 bg-gray-200 [&>div]:bg-fuchsia-600" 
          />
          <p className="text-sm text-gray-600 text-center">
            {mediaType === 'audio' ? progressText : processPhase || 'Processing translations...'}
          </p>
        </div>
      )}
    </div>
  );
};

export default TranslationFlow;