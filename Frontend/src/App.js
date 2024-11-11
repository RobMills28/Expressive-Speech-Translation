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
  const [targetLanguage, setTargetLanguage] = useState('fra');
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

  const handleLanguageSelect = (value) => {
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

    // Track cleanup tasks
    const cleanupTasks = [];
    let progressInterval;

    try {
        // Set initial states
        setProcessing(true);
        setError('');
        setProgress(10);
        setAudioStatus('loading');
        setAudioReady(false);

        // Clean up existing audio resources first
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
            setTranslatedAudioUrl('');
        }

        if (audioRef.current) {
            const oldSrc = audioRef.current.src;
            audioRef.current.pause();
            audioRef.current.src = '';
            audioRef.current.load();
            if (oldSrc && oldSrc.startsWith('blob:')) {
                URL.revokeObjectURL(oldSrc);
            }
        }

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_language', targetLanguage);

        // Create progress update interval
        progressInterval = setInterval(() => {
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
        cleanupTasks.push(() => clearInterval(progressInterval));

        // Set initial progress message
        setProgressText(getProgressMessage(10));

        // Make the API request
        const response = await fetch('http://localhost:5001/translate', {
          method: 'POST',
          body: formData,
          credentials: 'include',
          mode: 'cors',
          headers: {
              'Accept': '*/*',  // Change this to accept any content type
              'Origin': 'http://localhost:3000'
          }
      });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        // Validate response headers
        const contentType = response.headers.get('Content-Type');
        if (!contentType || !contentType.includes('audio/')) {
            throw new Error(`Server returned invalid content type: ${contentType}`);
        }

        // Get and validate audio data
        const audioData = await response.arrayBuffer();
        if (!audioData || audioData.byteLength === 0) {
            throw new Error('Server returned empty audio data');
        }

        const audioBlob = new Blob([audioData], { 
          type: 'audio/wav'  // Explicitly set the MIME type
      });
      
      // Log blob details
      console.log('Audio blob:', {
          size: audioBlob.size,
          type: audioBlob.type
      });
      
      // Create object URL with explicit type
      const audioUrl = URL.createObjectURL(audioBlob);
      console.log('Created URL:', audioUrl);

        if (audioBlob.size === 0) {
            throw new Error('Created audio blob is empty');
        }

        // Log audio data details
        console.log('Audio data received:', {
            size: audioBlob.size,
            type: audioBlob.type,
            byteLength: audioData.byteLength,
            contentType,
            headers: Object.fromEntries(response.headers.entries())
        });

        // Validate audio playability
        const isPlayable = await validateAudio(audioBlob);
        if (!isPlayable) {
            throw new Error('Audio validation failed - file may be corrupted');
        }

        // Create final URL after validation
        const finalUrl = URL.createObjectURL(audioBlob);
        cleanupTasks.push(() => URL.revokeObjectURL(finalUrl));

        // Update audio element
        if (audioRef.current) {
            audioRef.current.src = finalUrl;
            audioRef.current.load();
        }

        setTranslatedAudioUrl(finalUrl);
        
        // Set completion states
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
        
        console.error('Translation error:', e);
        setError(`Translation failed: ${e.message}`);
        setAudioStatus('error');
        setAudioReady(false);
        
        // Clean up any partially created resources
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
            setTranslatedAudioUrl('');
        }
    } finally {
        // Clean up all resources
        cleanupTasks.forEach(task => {
            try {
                task();
            } catch (e) {
                console.error('Cleanup task failed:', e);
            }
        });
        
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        setProcessing(false);
        abortControllerRef.current = null;
    }
};

// Helper function to validate audio
const validateAudio = async (audioBlob) => {
    return new Promise((resolve, reject) => {
        const testAudio = new Audio();
        const tempUrl = URL.createObjectURL(audioBlob);
        let timeoutId;

        const cleanup = () => {
            if (timeoutId) clearTimeout(timeoutId);
            testAudio.removeEventListener('canplaythrough', onCanPlay);
            testAudio.removeEventListener('error', onError);
            testAudio.src = '';
            testAudio.remove();
            URL.revokeObjectURL(tempUrl);
        };

        const onCanPlay = () => {
            if (testAudio.duration === 0 || isNaN(testAudio.duration)) {
                cleanup();
                resolve(false);
                return;
            }
            cleanup();
            resolve(true);
        };

        const onError = (e) => {
            console.error('Audio validation error:', {
                error: e.target.error,
                code: e.target.error?.code,
                message: e.target.error?.message,
                state: testAudio.readyState
            });
            cleanup();
            resolve(false);
        };

        timeoutId = setTimeout(() => {
            cleanup();
            resolve(false);
        }, 5000);

        testAudio.addEventListener('canplaythrough', onCanPlay, { once: true });
        testAudio.addEventListener('error', onError, { once: true });
        
        testAudio.src = tempUrl;
        testAudio.load();
    });
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
             defaultValue="fra"
             value={targetLanguage} 
             onValueChange={handleLanguageSelect}
           >
             <SelectTrigger className="w-full border-fuchsia-300">
               <SelectValue />
             </SelectTrigger>
             <SelectContent>
               <SelectItem value="fra">🇫🇷 French</SelectItem>
               <SelectItem value="deu">🇩🇪 German</SelectItem>
               <SelectItem value="ita">🇮🇹 Italian</SelectItem>
               <SelectItem value="por">🇵🇹 Portuguese</SelectItem>
               <SelectItem value="spa">🇪🇸 Spanish</SelectItem>
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
          setError(''); // Clear any previous errors
        }}
        onLoadedMetadata={(e) => {
          console.log('Audio metadata loaded', {
            duration: e.target.duration,
            src: e.target.src
          });
          if (e.target.duration === 0 || isNaN(e.target.duration)) {
            setError('Invalid audio duration');
            setAudioStatus('error');
            return;
          }
          if (audioRef.current) {
            console.log('Duration:', audioRef.current.duration);
          }
        }}
        onLoadedData={(e) => {
          console.log('Audio loaded successfully', {
            duration: e.target.duration,
            readyState: e.target.readyState,
            networkState: e.target.networkState,
            src: e.target.src
          });
          
          // Validate audio data
          if (!e.target.src || e.target.src === '') {
            setError('Audio source not available');
            setAudioStatus('error');
            return;
          }
          
          if (e.target.readyState < 2) { // HAVE_CURRENT_DATA
            setError('Audio data not fully loaded');
            setAudioStatus('error');
            return;
          }
          
          setProgress(100);
          setAudioStatus('ready');
          setAudioReady(true);
          setError(''); // Clear any errors if successful
        }}
        onCanPlay={() => {
          console.log('Audio can play');
          setAudioReady(true);
          setError(''); // Clear any errors
        }}
        onPlaying={() => {
          console.log('Audio playing');
          setIsPlaying(true);
          setAudioStatus('playing');
          setError(''); // Clear any errors
        }}
        onEnded={() => {
          console.log('Audio playback ended');
          setIsPlaying(false);
          setAudioStatus('ready');
        }}
        onError={(e) => {
          const error = e.target.error;
          const errorCode = error?.code;
          const errorMessage = error?.message || 'Unknown error';
          
          // Detailed error logging
          console.error('Audio error details:', {
            error,
            code: errorCode,
            message: errorMessage,
            src: e.target.src,
            readyState: audioRef.current?.readyState,
            networkState: audioRef.current?.networkState,
            currentSrc: e.target.currentSrc,
            audio: {
              duration: audioRef.current?.duration,
              paused: audioRef.current?.paused,
              muted: audioRef.current?.muted,
              volume: audioRef.current?.volume
            }
          });
        
          // User-friendly error messages based on error code
          let userMessage = 'An error occurred while playing the audio. ';
          switch (errorCode) {
            case 1: // MEDIA_ERR_ABORTED
              userMessage += 'The audio playback was interrupted.';
              break;
            case 2: // MEDIA_ERR_NETWORK
              userMessage += 'A network error occurred while loading the audio.';
              break;
            case 3: // MEDIA_ERR_DECODE
              userMessage += 'The audio file is corrupted or format is not supported.';
              break;
            case 4: // MEDIA_ERR_SRC_NOT_SUPPORTED
              userMessage += 'The audio format is not supported or the file is missing.';
              break;
            default:
              userMessage += errorMessage;
          }
        
          // Update state with error info
          setError(userMessage);
          setAudioStatus('error');
          setAudioReady(false);
          setIsPlaying(false);
        
          // Optional: Try to recover from error
          if (audioRef.current) {
            audioRef.current.load(); // Attempt to reload the audio
          }
        }}
        onPlay={() => {
          if (!audioReady) {
            console.warn('Attempting to play before audio is ready');
            setError('Audio is not ready to play yet');
            return;
          }
          console.log('Play requested');
          setIsPlaying(true);
          setAudioStatus('playing');
          setError(''); // Clear any errors
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
        onStalled={() => {
          console.log('Audio playback stalled');
          setAudioStatus('error');
          setError('Audio playback stalled. Please try again.');
        }}
        onSuspend={() => {
          console.log('Audio loading suspended');
          if (audioRef.current?.readyState < 2) {
            setAudioStatus('error');
            setError('Audio loading suspended. Please check your connection.');
          }
        }}
      />
    </div>

    {/* Status indicator
    {audioStatus === 'error' && error && (
      <Alert variant="destructive" className="mt-2">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )} */}

    {/* Play/Pause Button with enhanced feedback */}
    <Button
      onClick={handlePlayPause}
      className={`w-full ${
        isPlaying 
          ? 'bg-fuchsia-600 hover:bg-fuchsia-700' 
          : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700'
      } text-white transition-all duration-300`}
      disabled={!audioReady || audioStatus === 'error'}
      title={
        !audioReady 
          ? 'Audio is loading...' 
          : audioStatus === 'error' 
            ? 'Cannot play due to an error' 
            : isPlaying 
              ? 'Pause translation' 
              : 'Play translation'
      }
    >
      <div className="flex items-center justify-center">
        {audioStatus === 'loading' ? (
          <>
            <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
            <span>Loading audio...</span>
          </>
        ) : audioStatus === 'error' ? (
          <>
            <AlertCircle className="mr-2" size={18} />
            <span>Audio Unavailable</span>
          </>
        ) : isPlaying ? (
          <>
            <Pause className="mr-2" size={18} />
            <span>Pause Translation</span>
          </>
        ) : (
          <>
            <Play className="mr-2" size={18} />
            <span>Play Translation</span>
          </>
        )}
      </div>
    </Button>

{/* Optional: Add a retry button when in error state */}
{audioStatus === 'error' && (
  <Button
    onClick={() => {
      if (audioRef.current) {
        audioRef.current.load();
        setError('');
        setAudioStatus('loading');
      }
    }}
    variant="outline"
    className="w-full mt-2"
  >
    <div className="flex items-center justify-center">
      <div className="mr-2">↺</div>
      <span>Retry</span>
    </div>
  </Button>
)}
</div>
)}

{/* Translate Button */}
<Button 
  onClick={processAudio} 
  disabled={!file || processing}
  className={`
    w-full 
    mt-4
    text-white 
    transition-all 
    duration-200 
    ${!file
      ? 'bg-gradient-to-r from-fuchsia-300 to-pink-300 cursor-not-allowed'
      : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 cursor-pointer'
    }
    ${processing ? 'cursor-not-allowed hover:cursor-not-allowed' : ''}
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
</CardContent>
</Card>
</div>
  );
};

export default LinguaSyncApp;