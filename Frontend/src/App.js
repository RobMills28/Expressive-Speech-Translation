import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Label } from "./components/ui/label";
import { Input } from "./components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Progress } from "./components/ui/progress";
import { AlertCircle, Globe, Mic, Play, Pause, Upload, Link as LinkIcon, Clock } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import TranscriptView from "./components/ui/TranscriptView";
import Dashboard from "./components/Dashboard";
import { useTranslation } from './hooks/useTranslation';

// Navigation Component
const Navigation = () => {
  const location = useLocation();

  // Don't show navigation on landing page
  if (location.pathname === '/') {
    return null;
  }
  
  return (
    <nav className="bg-white border-b">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-fuchsia-600">
              Magenta AI
            </Link>
          </div>
          
          <div className="flex items-center gap-4">
            <Link 
              to="/translate"
              className={`px-4 py-2 text-gray-600 hover:text-gray-900 ${
                location.pathname === '/translate' ? 'text-fuchsia-600' : ''
              }`}
            >
              Translation Tool
            </Link>
            <Link 
              to="/dashboard"
              className={`px-4 py-2 rounded-lg hover:bg-fuchsia-700 ${
                location.pathname === '/dashboard' 
                  ? 'bg-fuchsia-700 text-white'
                  : 'bg-fuchsia-600 text-white'
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

// Landing Page Component
const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-fuchsia-50 to-white">
      <nav className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-fuchsia-600">Magenta AI</h1>
          <div className="flex items-center gap-4">
            <Link
              to="/translate"
              className="px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              Translation Tool
            </Link>
            <Link
              to="/dashboard"
              className="px-4 py-2 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 transition-colors"
            >
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto pt-32">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-600 to-pink-600">
            Share Your Content With the World, In Any Language
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Magenta helps creators reach global audiences by translating content while 
            preserving their authentic voice and style.
          </p>
          <Link
            to="/translate"
            className="inline-block px-8 py-4 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 text-lg shadow-lg hover:shadow-xl transition-all duration-200"
          >
            Get Started Free
          </Link>
        </div>

        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-1/2 left-1/2 -translate-x-1/2 w-full max-w-4xl h-[800px] rounded-full bg-fuchsia-100/50 blur-3xl" />
          <div className="absolute top-1/4 right-1/4 w-[600px] h-[600px] rounded-full bg-pink-100/30 blur-3xl" />
        </div>

        <div className="max-w-6xl mx-auto mt-16 px-4">
          <div className="aspect-video w-full max-w-3xl mx-auto rounded-xl bg-white/50 shadow-lg backdrop-blur-sm border border-white/20" />
        </div>
      </div>
    </div>
  );
};

// Translation View Component
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

  const [isRecording, setIsRecording] = React.useState(false);
  const [linkUrl, setLinkUrl] = React.useState('');
  const fileInputRef = React.useRef(null);

  useEffect(() => {
    const handleBeforeUnload = () => cleanup();
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      cleanup();
    };
  }, [cleanup]);

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      const event = { target: { files: [droppedFile] } };
      handleFileChange(event);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-fuchsia-700 via-fuchsia-500 to-pink-500 p-6">
      <div className="max-w-xl mx-auto">
        <Card className="shadow-xl bg-white/90 backdrop-blur-md">
          <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg">
            <CardTitle className="flex items-center justify-between">
              <span className="text-2xl font-bold">Quick Translate</span>
              <div className="text-sm font-normal text-pink-100 flex items-center gap-2">
                <Clock size={16} />
                2 minutes free
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="mt-4 space-y-6">
            <Tabs defaultValue="upload" className="space-y-6">
              <TabsList className="grid w-full grid-cols-3 gap-4 bg-fuchsia-50">
                <TabsTrigger value="upload" className="flex items-center gap-2">
                  <Upload size={16} />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="link" className="flex items-center gap-2">
                  <LinkIcon size={16} />
                  Paste Link
                </TabsTrigger>
                <TabsTrigger value="record" className="flex items-center gap-2">
                  <Mic size={16} />
                  Record
                </TabsTrigger>
              </TabsList>

              {/* File Upload Tab */}
              <TabsContent value="upload">
                <div
                  className="border-2 border-dashed border-fuchsia-200 rounded-lg p-8 text-center hover:border-fuchsia-400 transition-colors cursor-pointer bg-fuchsia-50/50"
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileChange}
                    accept="audio/*"
                    className="hidden"
                  />
                  <Upload size={32} className="mx-auto mb-4 text-fuchsia-400" />
                  <p className="text-sm text-fuchsia-800 font-medium">
                    {file ? file.name : "Drop your audio file here"}
                  </p>
                  <p className="text-xs text-fuchsia-600 mt-1">
                    {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "or click to browse"}
                  </p>
                  <p className="text-xs text-fuchsia-500/70 mt-4">
                    Supports MP3, WAV, M4A • Max 2 minutes
                  </p>
                </div>
              </TabsContent>

              {/* Link Input Tab */}
              <TabsContent value="link">
                <div className="space-y-2">
                  <Input
                    type="url"
                    placeholder="Paste YouTube, Spotify, or audio URL"
                    value={linkUrl}
                    onChange={(e) => setLinkUrl(e.target.value)}
                    className="w-full"
                    disabled
                  />
                  <p className="text-xs text-fuchsia-600">
                    Coming soon: YouTube videos, Spotify tracks, direct audio links
                  </p>
                </div>
              </TabsContent>

              {/* Record Tab */}
              <TabsContent value="record">
                <div className="border-2 border-fuchsia-200 rounded-lg p-8 text-center bg-fuchsia-50/50">
                  <Button
                    variant={isRecording ? "destructive" : "outline"}
                    className="gap-2"
                    onClick={() => setIsRecording(!isRecording)}
                    disabled
                  >
                    <Mic size={16} />
                    {isRecording ? "Stop Recording" : "Start Recording"}
                  </Button>
                  <p className="text-xs text-fuchsia-600 mt-4">
                    Coming soon: Record directly in your browser
                  </p>
                </div>
              </TabsContent>

              {/* Common Elements */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-fuchsia-800 flex items-center">
                    <Globe className="mr-2" size={18} />
                    Target Language
                  </Label>
                  <Select value={targetLanguage} onValueChange={handleLanguageSelect}>
                    <SelectTrigger className="w-full border-fuchsia-200">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fra">🇫🇷 French</SelectItem>
                      <SelectItem value="spa">🇪🇸 Spanish</SelectItem>
                      <SelectItem value="deu">🇩🇪 German</SelectItem>
                      <SelectItem value="ita">🇮🇹 Italian</SelectItem>
                      <SelectItem value="por">🇵🇹 Portuguese</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Error Display */}
                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

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

                {/* Translation Results */}
                {translatedAudioUrl && (
                  <div className="space-y-4">
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
                    text-white 
                    transition-all 
                    duration-200 
                    ${!file
                      ? 'bg-gradient-to-r from-fuchsia-300 to-pink-300 cursor-not-allowed'
                      : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700'
                    }
                    ${processing ? 'cursor-not-allowed' : ''}
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

                <div className="bg-fuchsia-50 p-4 rounded-lg">
                  <p className="text-sm text-fuchsia-900 font-medium">
                    Want more translation time?
                  </p>
                  <p className="text-sm text-fuchsia-700 mt-1">
                    Create a free account to get 5 additional minutes.
                  </p>
                  <Button 
                    variant="link" 
                    className="text-fuchsia-600 p-0 h-auto mt-2"
                  >
                    Create Account
                  </Button>
                </div>
              </div>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-white">
        <Navigation />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/translate" element={<TranslationView />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;