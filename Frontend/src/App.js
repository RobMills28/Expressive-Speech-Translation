import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card"
import { Button } from "./components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Label } from "./components/ui/label"
import { Input } from "./components/ui/input"
import { AlertCircle, Globe, Mic, Play, Pause } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert"
import { Progress } from "./components/ui/progress"

const LinguaSyncApp = () => {
  const [audioStatus, setAudioStatus] = useState('idle');
  const [audioReady, setAudioReady] = useState(false);
  const [file, setFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [progressText, setProgressText] = useState('');
  const audioRef = useRef(null);
  const abortControllerRef = useRef(null);

  const getProgressMessage = (progress) => {
    if (progress < 20) return "Preparing your audio for translation...";
    if (progress < 40) return "Analyzing speech patterns...";
    if (progress < 60) return "Converting to target language...";
    if (progress < 80) return "Generating natural speech...";
    if (progress < 100) return "Finalizing your translation...";
    return "Translation complete!";
  };

  useEffect(() => {
    // Define cleanup function
    const cleanup = () => {
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
        }
        if (audioRef.current) {
            audioRef.current.src = '';
            audioRef.current.load();
        }
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
    };

    // Add event listener
    window.addEventListener('beforeunload', cleanup);

    // Return cleanup function
    return () => {
        window.removeEventListener('beforeunload', cleanup);
        cleanup();
    };
}, [translatedAudioUrl]); // Added translatedAudioUrl as dependency

  const handleFileChange = (event) => {
    setError('');
    setProgress(0);
    setProgressText('');
    setAudioStatus('idle');
    
    if (translatedAudioUrl) {
      URL.revokeObjectURL(translatedAudioUrl);
      setTranslatedAudioUrl('');
    }

    const selectedFile = event.target.files[0];
    if (selectedFile) {
      console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
      
      const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      
      const validMimeTypes = [
        'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/wave',
        'audio/x-wav', 'audio/ogg', 'audio/x-m4a', 'audio/mp4', 'audio/aac'
      ];

      if (!validExtensions.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${validExtensions.join(', ')}`);
        setFile(null);
        return;
      }

      if (!validMimeTypes.includes(selectedFile.type) && selectedFile.type !== '') {
        console.warn('File MIME type:', selectedFile.type);
        console.warn('Valid MIME types:', validMimeTypes);
      }

      setFile(selectedFile);
    } else {
      setFile(null);
    }
  };

  const handleLanguageChange = (value) => {
    setError('');
    setProgress(0);
    setProgressText('');
    setAudioStatus('idle');
    
    if (translatedAudioUrl) {
      URL.revokeObjectURL(translatedAudioUrl);
      setTranslatedAudioUrl('');
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    setIsPlaying(false);
    setTargetLanguage(value);
  };

  const handlePlayPause = async () => {
    if (!audioRef.current) {
      console.error('No audio element reference');
      return;
    }

    try {
      if (audioRef.current.paused) {
        setAudioStatus('loading');
        const playAttempt = audioRef.current.play();
        
        if (playAttempt !== undefined) {
          await playAttempt;
          setIsPlaying(true);
          setAudioStatus('playing');
        }
      } else {
        audioRef.current.pause();
        setIsPlaying(false);
        setAudioStatus('ready');
      }
    } catch (error) {
      console.error('Playback error:', {
        error,
        audioState: {
          currentTime: audioRef.current.currentTime,
          duration: audioRef.current.duration,
          readyState: audioRef.current.readyState,
          networkState: audioRef.current.networkState,
          src: audioRef.current.currentSrc
        }
      });
      setError(`Playback failed: ${error.message}`);
      setAudioStatus('error');
      setIsPlaying(false);
    }
  };

  const processAudio = async () => {
    // Create new abort controller for this request
    if (abortControllerRef.current) {
        abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
        // Set initial states
        setProcessing(true);
        setError('');
        setProgress(10);
        setAudioStatus('loading');
        setAudioReady(false);

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_language', targetLanguage);

        // Create progress update interval
        const progressInterval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 90) {
                    clearInterval(progressInterval);
                    return prev;
                }
                const increment = Math.random() * 15;
                const newProgress = Math.min(prev + increment, 90);
                setProgressText(getProgressMessage(newProgress));
                return newProgress;
            });
        }, 2000);

        // Set initial progress message
        setProgressText(getProgressMessage(10));
      
        const response = await fetch('http://localhost:5001/translate', {
            method: 'POST',
            body: formData,
            signal: abortControllerRef.current.signal,
            headers: {
                'Accept': 'audio/wav'
            }
        });

        if (!response.ok) {
            clearInterval(progressInterval);
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        // Get the full response data as arrayBuffer
        const audioData = await response.arrayBuffer();
        if (!audioData || audioData.byteLength === 0) {
            throw new Error('Received empty audio data from server');
        }

        // Create blob with explicit MIME type from response
        const contentType = response.headers.get('Content-Type');
        if (!contentType || !contentType.includes('audio/')) {
            throw new Error(`Invalid content type: ${contentType}`);
        }

        const audioBlob = new Blob([audioData], { 
            type: contentType
        });

        console.log('Audio data:', {
            size: audioBlob.size,
            type: audioBlob.type,
            byteLength: audioData.byteLength,
            contentType: contentType,
            headers: Object.fromEntries(response.headers.entries())
        });

        // Clean up existing resources
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
        }

        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
            audioRef.current.src = '';
            audioRef.current.load();
        }

        // Test audio playability with better error handling
        await new Promise((resolve, reject) => {
            const testAudio = new Audio();
            const tempUrl = URL.createObjectURL(audioBlob);
            
            const cleanup = () => {
                testAudio.src = '';
                testAudio.remove();
                URL.revokeObjectURL(tempUrl);
            };
            
            const timeoutId = setTimeout(() => {
                cleanup();
                reject(new Error('Audio loading timed out'));
            }, 5000);
            
            testAudio.addEventListener('canplaythrough', () => {
                if (testAudio.duration === 0 || isNaN(testAudio.duration)) {
                    clearTimeout(timeoutId);
                    cleanup();
                    reject(new Error('Invalid audio duration'));
                    return;
                }
                clearTimeout(timeoutId);
                cleanup();
                resolve();
            }, { once: true });
            
            testAudio.addEventListener('error', (e) => {
                clearTimeout(timeoutId);
                cleanup();
                console.error('Audio validation error:', {
                    error: e.target.error,
                    code: e.target.error?.code,
                    message: e.target.error?.message,
                    state: testAudio.readyState
                });
                reject(new Error(`Audio validation failed: ${e.target.error?.message || 'Unknown error'} (code: ${e.target.error?.code || 'unknown'})`));
            }, { once: true });

            testAudio.src = tempUrl;
            testAudio.load();
        });

        // Create final URL after validation
        const finalUrl = URL.createObjectURL(audioBlob);
        setTranslatedAudioUrl(finalUrl);
        
        // Clear interval and set completion states
        clearInterval(progressInterval);
        setProgress(100);
        setProgressText('Translation complete! 🎉');
        setAudioStatus('ready');
        setIsPlaying(false);
        setAudioReady(true);

    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('Request aborted');
            return;
        }
        console.error('Error details:', e);
        setError(`Translation failed: ${e.message}`);
        setAudioStatus('error');
        setAudioReady(false);
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
            setTranslatedAudioUrl('');
        }
    } finally {
        setProcessing(false);
        abortControllerRef.current = null;
    }
};

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <Card className="w-[400px] bg-white/90 backdrop-blur-md shadow-xl">
        <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg">
          <CardTitle className="text-3xl font-bold">LinguaSync AI</CardTitle>
          <CardDescription className="text-pink-100">Audio Translation</CardDescription>
        </CardHeader>
        <CardContent className="mt-4 space-y-6">
          <div className="bg-fuchsia-100 p-4 rounded-lg shadow-inner">
            <Label htmlFor="audio-input" className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Mic className="mr-2" size={18} />
              Upload Audio
            </Label>
            <div className="relative mt-1">
              <Input 
                id="audio-input" 
                type="file" 
                accept=".mp3,.wav,.ogg,.m4a,audio/*" 
                onChange={handleFileChange} 
                className="hidden"
              />
              <label 
                htmlFor="audio-input" 
                className="cursor-pointer inline-flex items-center px-4 py-2 bg-white border border-fuchsia-300 rounded-md font-semibold text-xs text-fuchsia-700 uppercase tracking-widest shadow-sm hover:bg-fuchsia-50 focus:outline-none focus:border-fuchsia-300 focus:ring focus:ring-fuchsia-200 active:bg-fuchsia-100 disabled:opacity-25 transition"
              >
                Choose file
              </label>
              <span className="ml-3 text-sm text-fuchsia-600">
                {file ? file.name : "No file chosen"}
              </span>
            </div>
          </div>
          
          <div className="space-y-2">
            <Label className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Globe className="mr-2" size={18} />
              Target Language
            </Label>
            <Select 
              value={targetLanguage} 
              onValueChange={handleLanguageChange}
            >
              <SelectTrigger className="w-full border-fuchsia-300">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="deu">🇩🇪 German</SelectItem>
                <SelectItem value="fra">🇫🇷 French</SelectItem>
                <SelectItem value="spa">🇪🇸 Spanish</SelectItem>
                <SelectItem value="ita">🇮🇹 Italian</SelectItem>
                <SelectItem value="por">🇵🇹 Portuguese</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Progress indicator */}
          {processing && (
            <div className="space-y-2 p-4">
              <Progress 
              value={Math.round(progress)}
              className="w-full h-2 bg-fuchsia-100" 
              />
              <p className="text-center text-sm text-fuchsia-800 animate-pulse">
                {progressText}
              </p>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

{translatedAudioUrl && (
  <div className="mt-4 space-y-4">
    <div className="w-full bg-fuchsia-50 rounded-lg p-4">
      <audio 
        ref={audioRef} 
        src={translatedAudioUrl}
        className="hidden"
        preload="auto"
        onLoadStart={() => {
          console.log('Audio loading started');
          setAudioReady(false);
          setAudioStatus('loading');
        }}
        onLoadedMetadata={() => {
          console.log('Audio metadata loaded');
          if (audioRef.current) {
            console.log('Duration:', audioRef.current.duration);
          }
        }}
        onLoadedData={() => {
          console.log('Audio loaded successfully');
          if (audioRef.current) {
            console.log('Audio duration:', audioRef.current.duration);
            console.log('Audio ready state:', audioRef.current.readyState);
          }
          setProgress(100);
          setAudioStatus('ready');
          setAudioReady(true);
        }}
        onCanPlay={() => {
          console.log('Audio can play');
          setAudioReady(true);
        }}
        onPlaying={() => {
          console.log('Audio playing');
          setIsPlaying(true);
          setAudioStatus('playing');
        }}
        onEnded={() => {
          console.log('Audio playback ended');
          setIsPlaying(false);
          setAudioStatus('ready');
        }}
        onError={(e) => {
          const errorDetail = e.target.error?.message || 'Unknown error';
          const errorCode = e.target.error?.code;
          console.error('Audio error:', {
            error: e.target.error,
            code: errorCode,
            message: errorDetail,
            src: e.target.src,
            readyState: audioRef.current?.readyState,
            networkState: audioRef.current?.networkState
          });
          setError(`Error playing audio (${errorCode}): ${errorDetail}`);
          setAudioStatus('error');
          setAudioReady(false);
        }}
        onPlay={() => {
          if (!audioReady) {
            console.warn('Attempting to play before audio is ready');
            return;
          }
          console.log('Play requested');
          setIsPlaying(true);
          setAudioStatus('playing');
        }}
        onPause={() => {
          console.log('Audio paused');
          setIsPlaying(false);
          setAudioStatus('ready');
        }}
        onWaiting={() => {
          console.log('Audio buffering');
          setAudioStatus('loading');
        }}
        onSeeking={() => {
          console.log('Audio seeking');
        }}
        onSeeked={() => {
          console.log('Audio seek completed');
        }}
      />
    </div>
    <Button
      onClick={handlePlayPause}
      className={`w-full ${isPlaying 
        ? 'bg-fuchsia-600 hover:bg-fuchsia-700' 
        : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700'
      } text-white transition-all duration-300`}
      disabled={!audioReady || audioStatus === 'error'}
    >
{isPlaying ? (
  <div className="flex items-center justify-center">
    {audioStatus === 'loading' ? (
      <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
    ) : (
      <Pause className="mr-2" size={18} />
    )}
    {audioStatus === 'loading' ? 'Loading...' : 'Pause Translation'}
  </div>
) : (
  <div className="flex items-center justify-center">
    <Play className="mr-2" size={18} />
    <span>Play Translation</span>
  </div>
)}
    </Button>
  </div>
    )}
    </CardContent>
    <CardFooter>
    <Button 
  onClick={processAudio} 
  disabled={!file || !targetLanguage || processing}
  className={`
    w-full 
    text-white 
    transition-all 
    duration-200 
    ${processing 
      ? 'bg-fuchsia-300 cursor-not-allowed'
      : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 cursor-pointer'
    }
  `}
>
  {processing ? (
    <div className="flex items-center justify-center">
      <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
      {progressText}
    </div>
  ) : (
    'Translate Audio'
  )}
</Button>
    </CardFooter>
</Card>
</div>
  );
};

export default LinguaSyncApp;