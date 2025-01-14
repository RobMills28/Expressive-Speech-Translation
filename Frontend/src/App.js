import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Label } from "./components/ui/label";
import { Input } from "./components/ui/input";
import TranscriptView from "./components/ui/TranscriptView";
import Dashboard from "./components/Dashboard";
import { AlertCircle, Globe, Mic, Play, Pause } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import { Progress } from "./components/ui/progress";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';  // Added useLocation back

const Navigation = () => {
  const location = useLocation();
  
  return (
    <nav className="bg-white border-b">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-purple-600">
              Magenta AI
            </Link>
          </div>
          
          <div className="flex items-center gap-4">
            <Link 
              to="/translate"
              className="px-4 py-2 text-gray-600 hover:text-gray-900"
            >
              Translation Tool
            </Link>
            <Link 
              to="/dashboard"
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

const App = () => {
  // All state
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
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [showTranscript, setShowTranscript] = useState(false);

  // Refs
  const audioRef = useRef(null);
  const abortControllerRef = useRef(null);
  const progressIntervalRef = useRef(null);

  const getProgressMessage = (progress) => {
    if (progress < 20) return "Preparing your audio for translation...";
    if (progress < 40) return "Analyzing speech patterns...";
    if (progress < 60) return "Converting to target language...";
    if (progress < 80) return "Generating natural speech...";
    if (progress < 100) return "Finalizing your translation...";
    return "Translation complete!";
  };

  const cleanup = useCallback(() => {
    try {
      if (translatedAudioUrl) {
        URL.revokeObjectURL(translatedAudioUrl);
        setTranslatedAudioUrl('');
      }
      
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        audioRef.current.removeAttribute('src');
        audioRef.current.load();
      }
      
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      
      setAudioReady(false);
      setAudioStatus('idle');
      setError('');
      setProgress(0);
      setProgressText('');
      setIsPlaying(false);
      setSourceText('');
      setTargetText('');
      setShowTranscript(false);
    } catch (e) {
      console.error('Cleanup error:', e);
    }
  }, [translatedAudioUrl]);

  useEffect(() => {
    window.addEventListener('beforeunload', cleanup);
    return () => {
      window.removeEventListener('beforeunload', cleanup);
      cleanup();
    };
  }, [cleanup]);

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
    cleanup();
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
    cleanup();
    abortControllerRef.current = new AbortController();

    try {
        setProcessing(true);
        setProgress(10);
        setProgressText(getProgressMessage(10));
        setAudioStatus('loading');

        progressIntervalRef.current = setInterval(() => {
            setProgress(prev => {
                if (prev >= 90) {
                    clearInterval(progressIntervalRef.current);
                    return prev;
                }
                const increment = Math.random() * 15;
                const newProgress = Math.min(prev + increment, 90);
                setProgressText(getProgressMessage(newProgress));
                return newProgress;
            });
        }, 2000);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_language', targetLanguage);

        const response = await fetch('http://localhost:5001/translate', {
            method: 'POST',
            body: formData,
            credentials: 'include',
            signal: abortControllerRef.current.signal
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const responseData = await response.json();
        const audioBuffer = Uint8Array.from(atob(responseData.audio), c => c.charCodeAt(0));
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        
        if (audioBlob.size === 0) {
            throw new Error('Received empty audio data');
        }

        const url = URL.createObjectURL(audioBlob);

        if (audioRef.current) {
            audioRef.current.src = url;
            await audioRef.current.load();
        }

        setSourceText(responseData.transcripts.source);
        setTargetText(responseData.transcripts.target);
        setTranslatedAudioUrl(url);
        setProgress(100);
        setProgressText(getProgressMessage(100));
        setAudioReady(true);
        setAudioStatus('ready');
        setShowTranscript(true);

    } catch (e) {
        console.error('Translation error:', e);
        setError(e.message);
        setAudioStatus('error');
        cleanup();
    } finally {
        setProcessing(false);
        if (progressIntervalRef.current) {
            clearInterval(progressIntervalRef.current);
            progressIntervalRef.current = null;
        }
    }
};

// Translation View Component
const TranslationView = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <Card className="w-[400px] bg-white/90 backdrop-blur-md shadow-xl">
        <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg">
          <CardTitle className="text-3xl font-bold">LinguaSync AI</CardTitle>
          <CardDescription className="text-pink-100">Audio Translation</CardDescription>
        </CardHeader>
        <CardContent className="mt-4 space-y-6 pb-6">
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
              <div>
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
                  onCanPlayThrough={() => {
                    console.log('Audio can play through');
                    setAudioStatus('ready');
                    setAudioReady(true);
                    setError('');
                  }}
                  onPlaying={() => {
                    console.log('Audio is playing');
                    setIsPlaying(true);
                    setAudioStatus('playing');
                  }}
                  onEnded={() => {
                    console.log('Audio playback ended');
                    setIsPlaying(false);
                    setAudioStatus('ready');
                  }}
                  onError={(e) => {
                    console.error('Audio error:', e.target.error);
                    setError('Failed to load audio');
                    setAudioStatus('error');
                    setAudioReady(false);
                    setIsPlaying(false);
                  }}
                  onPause={() => {
                    console.log('Audio paused');
                    setIsPlaying(false);
                    setAudioStatus('ready');
                  }}
                />
              </div>

              <Button
                onClick={handlePlayPause}
                className={`w-full ${
                  isPlaying 
                    ? 'bg-fuchsia-600 hover:bg-fuchsia-700' 
                    : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700'
                } text-white transition-all duration-300`}
                disabled={!audioReady || audioStatus === 'error'}
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

              {showTranscript && (
                <TranscriptView 
                  sourceText={sourceText}
                  targetText={targetText}
                  sourceLang="English"
                  targetLang={targetLanguage}
                />
              )}

              {audioStatus === 'error' && (
                <Button
                  onClick={() => {
                    if (audioRef.current && translatedAudioUrl) {
                      audioRef.current.src = translatedAudioUrl;
                      audioRef.current.load();
                      setAudioStatus('loading');
                      setError('');
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

// Main App Return with Router
return (
  <Router>
    <div className="min-h-screen bg-white">
      <Navigation />
      <Routes>
        <Route path="/" element={<TranslationView />} />
        <Route path="/translate" element={<TranslationView />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </div>
  </Router>
);
};

export default App;