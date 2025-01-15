import React, { useEffect } from 'react';
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
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { useTranslation } from './hooks/useTranslation';

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
              className={`px-4 py-2 text-gray-600 hover:text-gray-900 ${
                location.pathname === '/translate' ? 'text-purple-600' : ''
              }`}
            >
              Translation Tool
            </Link>
            <Link 
              to="/dashboard"
              className={`px-4 py-2 rounded-lg hover:bg-purple-700 ${
                location.pathname === '/dashboard' 
                  ? 'bg-purple-700 text-white'
                  : 'bg-purple-600 text-white'
              }`}
            >
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

const TranslationView = () => {
  const {
    audioStatus,
    audioReady,
    file,
    targetLanguage,
    processing,
    progress,
    error,
    translatedAudioUrl,
    isPlaying,
    progressText,
    sourceText,
    targetText,
    showTranscript,
    audioRef,
    handleFileChange,
    handleLanguageSelect,
    handlePlayPause,
    processAudio,
    cleanup,
    setError,
    setAudioStatus,
    setAudioReady,
    setIsPlaying,
    setShowTranscript
  } = useTranslation();

  // Handle cleanup on component unmount and page refresh
  useEffect(() => {
    const handleBeforeUnload = () => cleanup();
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      cleanup();
    };
  }, [cleanup]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <Card className="w-[400px] bg-white/90 backdrop-blur-md shadow-xl">
        <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg">
          <CardTitle className="text-3xl font-bold">LinguaSync AI</CardTitle>
          <CardDescription className="text-pink-100">Audio Translation</CardDescription>
        </CardHeader>
        <CardContent className="mt-4 space-y-6 pb-6">
          {/* File Upload Section */}
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

          {/* Language Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Globe className="mr-2" size={18} />
              Target Language
            </Label>
            <Select 
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

          {/* Progress Indicator */}
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

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Translation Results and Audio Player */}
          {translatedAudioUrl && (
            <div className="mt-4 space-y-4">
              {/* Hidden Audio Element */}
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
                  console.error('Audio error:', e);
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

              {/* Play/Pause Button */}
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

              {/* Transcript Display */}
              {showTranscript && (
                <TranscriptView 
                  sourceText={sourceText}
                  targetText={targetText}
                  sourceLang="English"
                  targetLang={targetLanguage}
                />
              )}

              {/* Retry Button for Audio Errors */}
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

          {/* Translation Button */}
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

const App = () => {
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