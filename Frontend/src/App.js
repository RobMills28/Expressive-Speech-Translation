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

  // Cleanup audio URL on unmount
  useEffect(() => {
    return () => {
      if (translatedAudioUrl) {
        URL.revokeObjectURL(translatedAudioUrl);
      }
    };
  }, [translatedAudioUrl]);

  const handleFileChange = (event) => {
    // Clear previous state first
    setError('');
    setProgress(0);
    setProgressText('');
    
    // Clean up any existing audio URL
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
            'audio/mp3',
            'audio/mpeg',
            'audio/wav',
            'audio/wave',
            'audio/x-wav',
            'audio/ogg',
            'audio/x-m4a',
            'audio/mp4',
            'audio/aac'
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
  // Clear states first
  setError('');
  setProgress(0);
  setProgressText('');
  
  // Clean up any existing audio URL
  if (translatedAudioUrl) {
      URL.revokeObjectURL(translatedAudioUrl);
      setTranslatedAudioUrl('');
  }

  console.log('Language selected:', value);
  setTargetLanguage(value);
};

const handlePlayPause = async () => {
  if (!audioRef.current) {
      console.error('No audio element reference');
      return;
  }

  try {
      if (audioRef.current.paused) {
          // Create a new promise to handle the play attempt
          const playAttempt = audioRef.current.play();
          
          if (playAttempt !== undefined) {
              await playAttempt;
              setIsPlaying(true);
          }
      } else {
          audioRef.current.pause();
          setIsPlaying(false);
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
  }
};

const processAudio = async () => {
    try {
        setProcessing(true);
        setError('');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_language', targetLanguage);

        console.log('Sending request...');
        const response = await fetch('http://localhost:5001/translate', {
            method: 'POST',
            body: formData,
        });

        console.log('Response status:', response.status);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        // Create blob directly from response
        const blob = await response.blob();
        console.log('Response blob:', {
            size: blob.size,
            type: blob.type
        });

        // Clean up old URL
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
        }

        const url = URL.createObjectURL(blob);
        
        // Test audio playback before setting state
        await new Promise((resolve, reject) => {
            const audio = new Audio();
            audio.oncanplaythrough = resolve;
            audio.onerror = () => reject(new Error('Audio format not supported'));
            audio.src = url;
            
            // Set timeout in case loading hangs
            const timeout = setTimeout(() => {
                audio.src = '';
                reject(new Error('Audio loading timed out'));
            }, 5000);
            
            // Clear timeout if audio loads
            audio.oncanplaythrough = () => {
                clearTimeout(timeout);
                resolve();
            };
        });

        setTranslatedAudioUrl(url);
        setProgress(100);
        setProgressText('Translation complete!');

    } catch (e) {
        console.error('Error details:', e);
        setError(`An error occurred: ${e.message}`);
        if (translatedAudioUrl) {
            URL.revokeObjectURL(translatedAudioUrl);
            setTranslatedAudioUrl('');
        }
    } finally {
        setProcessing(false);
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

          {processing && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <p className="text-center text-sm text-fuchsia-800">{progressText}</p>
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
    <div className="mt-4 space-y-2">
        <div className="w-full bg-white rounded-lg p-2">
            <audio 
                ref={audioRef} 
                controls
                className="w-full"
                src={translatedAudioUrl}
                preload="auto"
                onLoadedData={() => {
                    console.log('Audio loaded successfully');
                    setProgress(100);
                }}
                onEnded={() => setIsPlaying(false)}
                onError={(e) => {
                    console.error('Audio error:', {
                        error: e.target.error,
                        code: e.target.error?.code,
                        message: e.target.error?.message
                    });
                    setError('Error playing audio: ' + (e.target.error?.message || 'Unknown error'));
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
            />
        </div>
        <Button
            onClick={() => {
                if (audioRef.current) {
                    if (audioRef.current.paused) {
                        audioRef.current.play().catch(error => {
                            console.error('Play error:', error);
                            setError('Failed to play audio: ' + error.message);
                        });
                    } else {
                        audioRef.current.pause();
                    }
                }
            }}
            className="w-full bg-fuchsia-100 text-fuchsia-800 hover:bg-fuchsia-200"
            disabled={!audioRef.current}
        >
            {isPlaying ? (
                <><Pause className="mr-2" size={18} /> Pause</>
            ) : (
                <><Play className="mr-2" size={18} /> Play Translation</>
            )}
        </Button>
    </div>
)}
        </CardContent>
        <CardFooter>
          <Button 
            onClick={processAudio} 
            disabled={!file || !targetLanguage || processing}
            className="w-full bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 text-white"
          >
            {processing ? progressText : 'Translate Audio'}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LinguaSyncApp;