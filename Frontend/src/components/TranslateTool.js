import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { Progress } from "./ui/progress";
import { AlertCircle, Globe, Mic, Play, Pause, Upload, Link as LinkIcon, Clock } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import TranscriptView from "./ui/TranscriptView";
import { LinkSection } from "./ui/LinkSection";
import { useTranslation } from '../hooks/useTranslation';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { useAudioLink } from '../hooks/useAudioLink';

const TranslateTool = () => {
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

  const {
    isRecording,
    recordedAudio,
    startRecording,
    stopRecording,
    clearRecording
  } = useAudioRecorder();

  const {
    isProcessing: isProcessingLink,
    error: linkError,
    processLink
  } = useAudioLink();

  const [linkUrl, setLinkUrl] = React.useState('');
  const fileInputRef = useRef(null);

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
          <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg p-6">
            <CardTitle className="flex items-center justify-between">
              <span className="text-2xl font-semibold tracking-tight">Quick Translate</span>
              <div className="text-sm font-normal text-pink-100 flex items-center gap-2 bg-white/10 px-3 py-1.5 rounded-full">
                <Clock size={14} />
                <span>2 minutes free</span>
              </div>
            </CardTitle>
          </CardHeader>

          <CardContent className="p-6">
            <Tabs defaultValue="upload" className="space-y-8">
              <TabsList className="grid w-full grid-cols-3 gap-2 p-1 bg-fuchsia-50 rounded-lg">
                <TabsTrigger 
                  value="upload" 
                  className="flex items-center gap-2 px-4 py-2.5 data-[state=active]:bg-white data-[state=active]:text-fuchsia-700 data-[state=active]:shadow-sm"
                >
                  <Upload size={16} />
                  <span className="font-medium">Upload</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="link"
                  className="flex items-center gap-2 px-4 py-2.5 data-[state=active]:bg-white data-[state=active]:text-fuchsia-700 data-[state=active]:shadow-sm"
                >
                  <LinkIcon size={16} />
                  <span className="font-medium">Link</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="record"
                  className="flex items-center gap-2 px-4 py-2.5 data-[state=active]:bg-white data-[state=active]:text-fuchsia-700 data-[state=active]:shadow-sm"
                >
                  <Mic size={16} />
                  <span className="font-medium">Record</span>
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="mt-6">
                <div
                  className="border-2 border-dashed border-fuchsia-200 rounded-xl p-8 text-center hover:border-fuchsia-400 transition-colors cursor-pointer bg-fuchsia-50/50"
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
                  <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mx-auto">
                    <Upload size={24} className="text-fuchsia-500" />
                  </div>
                  <div className="mt-4">
                    <p className="text-sm font-medium text-fuchsia-800">
                      {file ? file.name : "Drop your audio file here"}
                    </p>
                    <p className="text-xs text-fuchsia-600 mt-1">
                      {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "or click to browse"}
                    </p>
                  </div>
                  <div className="pt-4">
                    <p className="text-xs text-fuchsia-500/70 px-6 py-1.5 bg-white/50 rounded-full inline-block">
                      Supports MP3, WAV, M4A • Max 2 minutes
                    </p>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="link" className="mt-6">
                <LinkSection
                  linkUrl={linkUrl}
                  setLinkUrl={setLinkUrl}
                  isProcessingLink={isProcessingLink}
                  processLink={async (url) => {
                    try {
                      const result = await processLink(url);
                      if (result?.audioFile) {
                        // This line is key - it triggers the same handler used by file uploads
                        handleFileChange({ target: { files: [result.audioFile] } });
                      }
                    } catch (error) {
                      console.error('Failed to process link:', error);
                    }
                  }} 
                />
                {linkError && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{linkError}</AlertDescription>
                  </Alert>
                )}
              </TabsContent>
              <TabsContent value="record" className="mt-6">
                <div className="border-2 border-dashed border-fuchsia-200 rounded-xl p-8 text-center bg-fuchsia-50/50">
                  {recordedAudio ? (
                    <div className="space-y-4">
                      <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mx-auto">
                        <Play size={24} className="text-fuchsia-500" />
                      </div>
                      <p className="text-sm font-medium text-fuchsia-800">
                        Audio recorded! ({(recordedAudio.size / (1024 * 1024)).toFixed(2)} MB)
                      </p>
                      <div className="flex justify-center gap-2">
                        <Button
                          variant="outline"
                          className="gap-2"
                          onClick={() => {
                            handleFileChange({ target: { files: [recordedAudio] } });
                          }}
                        >
                          Use Recording
                        </Button>
                        <Button
                          variant="outline"
                          className="gap-2"
                          onClick={clearRecording}
                        >
                          Record Again
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="w-12 h-12 rounded-full bg-fuchsia-100 flex items-center justify-center mx-auto">
                        <Mic size={24} className="text-fuchsia-500" />
                      </div>
                      <Button
                        variant={isRecording ? "destructive" : "outline"}
                        className="gap-2 mt-4 inline-flex items-center justify-center"
                        onClick={isRecording ? stopRecording : startRecording}
                      >
                        <Mic size={16} className="text-fuchsia-500" />
                        {isRecording ? "Stop Recording" : "Start Recording"}
                      </Button>
                      {isRecording && (
                        <p className="text-sm text-fuchsia-600 animate-pulse mt-2">
                          Recording in progress...
                        </p>
                      )}
                    </>
                  )}
                </div>
              </TabsContent>

              <div className="space-y-6">
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

                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

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

                {translatedAudioUrl && (
                  <div className="space-y-4">
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

export default TranslateTool;