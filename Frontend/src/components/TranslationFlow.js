import React, { useState, useRef } from 'react';
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, Upload, Loader2, Film, AudioWaveform, Zap, Link, FileText } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";
import { Label } from "./ui/label";
import WaveformPlayer from './WaveformPlayer';
import { PlaceholderWaveform } from './PlaceholderWaveform';
import InputSelector from './InputSelector';

// --- THIS IS THE UPDATED LANGUAGE LIST ---
const LANGUAGES = {
  'eng': { name: 'English', flag: '🇬🇧' },
  'cmn': { name: 'Chinese (Mandarin)', flag: '🇨🇳' },
  'yue': { name: 'Cantonese', flag: '🇭🇰' },
  'jpn': { name: 'Japanese', flag: '🇯🇵' },
  'kor': { name: 'Korean', flag: '🇰🇷' },
  'fra': { name: 'French', flag: '🇫🇷' },
  'spa': { name: 'Spanish', flag: '🇪🇸' },
  'deu': { name: 'German', flag: '🇩🇪' },
  'ita': { name: 'Italian', flag: '🇮🇹' },
  'rus': { name: 'Russian', flag: '🇷🇺' },
  'ell': { name: 'Greek', flag: '🇬🇷' },
};

const ContentTranslator = () => {
  const [currentScreen, setCurrentScreen] = useState('selection');
  const [contentType, setContentType] = useState(null);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processPhase, setProcessPhase] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [resultTranscripts, setResultTranscripts] = useState({ source: '', target: '' });
  const [applyLipSync, setApplyLipSync] = useState(true);
  const [useVoiceCloningVideo, setUseVoiceCloningVideo] = useState(true);

  const originalMediaRef = useRef(null);
  const translatedMediaRef = useRef(null);

  const handleContentTypeSelect = (type) => {
    setContentType(type);
    setCurrentScreen('translator');
    setFile(null); setFileUrl(null); setResult(null);
    setResultTranscripts({ source: '', target: '' });
    setError(''); setProgress(0); setProcessPhase('');
    setTargetLanguage(type === 'audio' ? 'fra' : '');
    setApplyLipSync(true);
    setUseVoiceCloningVideo(true);
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    let valid = false;
    const isAudio = uploadedFile.type.startsWith('audio/');
    const isVideo = uploadedFile.type.startsWith('video/');

    if (contentType === 'audio' && (isAudio || isVideo)) valid = true;
    else if ((contentType === 'video' || contentType === 'both') && isVideo) valid = true;

    if (!valid) { setError(`Please upload a valid ${contentType} file.`); return; }
    if (uploadedFile.size > 150 * 1024 * 1024) { setError('File max 150MB.'); return; }

    setFile(uploadedFile);
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFileUrl(URL.createObjectURL(uploadedFile));
    setError(''); setResult(null); setResultTranscripts({ source: '', target: '' });
    setProgress(0); setProcessPhase('');
  };

  const handleTranslate = async () => {
    if (!file) { setError('Please upload a file.'); return; }
    if (!targetLanguage) { setError('Please select a target language.'); return; }

    setIsProcessing(true); setProgress(0); setProcessPhase('Preparing...'); setError(''); setResult(null); setResultTranscripts({ source: '', target: ''});

    const formData = new FormData();
    const fileKey = contentType === 'audio' ? 'file' : 'video';
    formData.append(fileKey, file);
    formData.append('target_language', targetLanguage);
    formData.append('backend', 'cascaded');

    if (contentType === 'video' || contentType === 'both') {
      formData.append('apply_lip_sync', applyLipSync ? 'true' : 'false');
      formData.append('use_voice_cloning', useVoiceCloningVideo ? 'true' : 'false');
    }

    const endpoint = contentType === 'audio' ? 'http://localhost:5001/translate' : 'http://localhost:5001/process-video';

    try {
      const response = await fetch(endpoint, { method: 'POST', body: formData });

      if (contentType === 'audio') {
        if (!response.ok) {
          const errData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
          throw new Error(errData.error || `Request failed: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const byteCharacters = atob(data.audio);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const audioBlob = new Blob([byteArray], { type: 'audio/wav' });
        
        if (result) URL.revokeObjectURL(result);
        setResult(URL.createObjectURL(audioBlob));
        
        setResultTranscripts(data.transcripts || { source: '', target: '' });
        setProgress(100);
        setProcessPhase('Completed!');
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            setProcessPhase(prev => error || prev.includes('Error') ? prev : 'Final video processed.');
            if(!error) setProgress(100);
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || '';
          for (const message of messages) {
            if (message.trim().startsWith('data: ')) {
              try {
                const data = JSON.parse(message.trim().slice(6));
                if (data.error) {
                    setError(data.error + (data.details ? `: ${data.details}` : ''));
                    setProcessPhase(`Error: ${data.phase || 'processing'}`);
                    setIsProcessing(false);
                    return;
                }
                if (data.progress !== undefined) setProgress(data.progress);
                if (data.phase) setProcessPhase(data.phase);
                if (data.result) {
                  const byteChars = atob(data.result);
                  const byteNums = new Array(byteChars.length);
                  for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
                  const byteArr = new Uint8Array(byteNums);
                  const videoBlob = new Blob([byteArr], { type: 'video/mp4' });
                  if (result) URL.revokeObjectURL(result);
                  setResult(URL.createObjectURL(videoBlob));
                  setProcessPhase('Video ready!');
                  setProgress(100);
                }
                if (data.transcripts) setResultTranscripts(data.transcripts);
              } catch (e) {
                  console.error('SSE parse error:', e, "Message:", message);
                  if (!error) setError("Error processing server response.");
              }
            }
          }
        }
      }
    } catch (err) {
        setError(err.message);
        console.error("Translate error:", err);
        setProcessPhase('Failed.');
    } finally {
        setIsProcessing(false);
    }
  };

  const handleReset = () => {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    if (result) URL.revokeObjectURL(result);
    setCurrentScreen('selection'); setContentType(null); setFile(null);
    setFileUrl(null); setTargetLanguage(''); setResult(null);
    setResultTranscripts({ source: '', target: '' }); setError('');
    setProgress(0); setProcessPhase('');
    setApplyLipSync(true);
    setUseVoiceCloningVideo(true);
  };

  const handleUrlSubmit = async (url) => {
    if (!url || !targetLanguage) {
      setError('Please select a language and enter a valid URL.');
      return;
    }
    setError('');
    setIsProcessing(true);
    setProgress(5);
    setProcessPhase('Processing from URL...');
    setResult(null); 
    setResultTranscripts({ source: '', target: '' });

    const endpoint = 'http://localhost:5001/process-audio-url';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url, target_language: targetLanguage, backend: 'cascaded' })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Failed to process URL.');
        }

        const data = await response.json();
        
        const byteCharacters = atob(data.audio);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const audioBlob = new Blob([byteArray], { type: 'audio/wav' });
        
        if (result) URL.revokeObjectURL(result);
        setResult(URL.createObjectURL(audioBlob));
        
        const mockFile = new File([audioBlob], "audio_from_url.wav", { type: "audio/wav" });
        setFile(mockFile);
        if (fileUrl) URL.revokeObjectURL(fileUrl);
        setFileUrl(URL.createObjectURL(audioBlob));

        setResultTranscripts(data.transcripts || { source: '', target: '' });
        setProgress(100);
        setProcessPhase('Completed!');
    } catch (err) {
        setError(err.message);
        setProcessPhase('Failed.');
    } finally {
        setIsProcessing(false);
    }
  };

  const renderSelectionScreen = () => (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-fuchsia-50/20 dark:to-fuchsia-950/10 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-fuchsia-500 to-pink-500 rounded-2xl mb-6 shadow-lg shadow-fuchsia-500/25">
            <AudioWaveform className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl md:text-5xl mb-4 bg-gradient-to-r from-foreground to-fuchsia-600 bg-clip-text text-transparent">
            Choose Your Content Type
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Transform your audio and video content with AI-powered translation that preserves emotion and voice characteristics
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <Card className="group relative overflow-hidden border-0 bg-background/60 backdrop-blur-sm hover:bg-gradient-to-br hover:from-background/80 hover:to-fuchsia-50/30 dark:hover:to-fuchsia-950/20 transition-all duration-500 cursor-pointer hover:shadow-2xl hover:shadow-fuchsia-500/10 hover:-translate-y-2"
                onClick={() => handleContentTypeSelect('audio')}>
            <div className="absolute inset-0 bg-gradient-to-br from-fuchsia-500/5 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative p-8 text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-fuchsia-100 to-pink-100 dark:from-fuchsia-900/30 dark:to-pink-900/30 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500">
                <AudioWaveform className="w-10 h-10 text-fuchsia-600 dark:text-fuchsia-400" />
              </div>
              <h3 className="text-xl mb-3 group-hover:text-fuchsia-600 dark:group-hover:text-fuchsia-400 transition-colors">Audio Translation</h3>
              <p className="text-muted-foreground mb-6">
                Perfect for podcasts, voice notes, interviews, and audio content. Preserve the speaker's emotional tone and vocal characteristics.
              </p>
              <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                <AudioWaveform className="w-4 h-4" />
                <span>Voice Preservation</span>
              </div>
            </div>
          </Card>

          <Card className="group relative overflow-hidden border-0 bg-background/60 backdrop-blur-sm hover:bg-gradient-to-br hover:from-background/80 hover:to-blue-50/30 dark:hover:to-blue-950/20 transition-all duration-500 cursor-pointer hover:shadow-2xl hover:shadow-blue-500/10 hover:-translate-y-2"
                onClick={() => handleContentTypeSelect('video')}>
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative p-8 text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500">
                <Film className="w-10 h-10 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">Video Translation</h3>
              <p className="text-muted-foreground mb-6">
                Translate video audio tracks while maintaining synchronization. Ideal for educational content, presentations, and media.
              </p>
              <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                <Film className="w-4 h-4" />
                <span>Audio Track Translation</span>
              </div>
            </div>
          </Card>

          <Card className="group relative overflow-hidden border-0 bg-background/60 backdrop-blur-sm hover:bg-gradient-to-br hover:from-background/80 hover:to-violet-50/30 dark:hover:to-violet-950/20 transition-all duration-500 cursor-pointer hover:shadow-2xl hover:shadow-violet-500/10 hover:-translate-y-2"
                onClick={() => handleContentTypeSelect('both')}>
            <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative p-8 text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-violet-100 to-purple-100 dark:from-violet-900/30 dark:to-purple-900/30 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500">
                <div className="relative">
                  <Film className="w-10 h-10 text-violet-600 dark:text-violet-400" />
                  <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
                    <Zap className="w-3 h-3 text-white" />
                  </div>
                </div>
              </div>
              <h3 className="text-xl mb-3 group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">Video Dubbing & Lip Sync</h3>
              <p className="text-muted-foreground mb-6">
                Complete video transformation with voice-over and lip synchronization. Perfect for professional dubbing and localization.
              </p>
              <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                <Zap className="w-4 h-4" />
                <span>Full Dubbing + Lip Sync</span>
              </div>
            </div>
          </Card>
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-muted-foreground">
            Powered by advanced AI models • Supports 10+ languages • Preserves emotional expression
          </p>
        </div>
      </div>
    </div>
  );

  const renderAudioInterface = () => (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-fuchsia-50/10 dark:to-fuchsia-950/5 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
            <div className="flex items-center space-x-4 mb-4 md:mb-0">
              <div className="w-12 h-12 bg-gradient-to-br from-fuchsia-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-fuchsia-500/25">
                <AudioWaveform className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl bg-gradient-to-r from-foreground to-fuchsia-600 bg-clip-text text-transparent">Audio Translation</h1>
                <p className="text-muted-foreground">Transform speech from audio or video files with emotion preservation</p>
              </div>
            </div>
            <Button variant="outline" onClick={handleReset} className="border-fuchsia-200 dark:border-fuchsia-800 hover:bg-fuchsia-50 dark:hover:bg-fuchsia-950/20">
              Change Type
            </Button>
          </div>
          
          {error && (
            <Alert variant="destructive" className="mb-6 border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-950/20">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Original Audio Panel */}
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-lg flex items-center justify-center">
                  <FileText className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                </div>
                <h3 className="text-lg">Original Audio</h3>
              </div>
              
              <div className="min-h-[280px] bg-gradient-to-br from-gray-50 to-gray-100/50 dark:from-gray-900/50 dark:to-gray-800/50 rounded-xl border border-gray-200/50 dark:border-gray-700/50 flex flex-col items-center justify-center p-6 transition-all duration-300">
                {fileUrl ? (
                  <div className="w-full">
                    <WaveformPlayer url={fileUrl} />
                    <div className="mt-4 p-3 bg-background/80 rounded-lg">
                      <p className="text-xs text-muted-foreground truncate">{file?.name}</p>
                    </div>
                  </div>
                ) : (
                  <InputSelector onFileChange={handleFileUpload} onUrlSubmit={handleUrlSubmit} contentType="audio" />
                )}
              </div>
            </div>
          </Card>

          {/* Translated Audio Panel */}
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-fuchsia-100 to-pink-100 dark:from-fuchsia-900/30 dark:to-pink-900/30 rounded-lg flex items-center justify-center">
                  <AudioWaveform className="w-4 h-4 text-fuchsia-600 dark:text-fuchsia-400" />
                </div>
                <h3 className="text-lg">
                  {targetLanguage && LANGUAGES[targetLanguage] 
                    ? `${LANGUAGES[targetLanguage].flag} ${LANGUAGES[targetLanguage].name} Translation` 
                    : 'Translated Audio'}
                </h3>
              </div>
              
              <div className="min-h-[280px] bg-gradient-to-br from-fuchsia-50/50 to-pink-50/50 dark:from-fuchsia-950/20 dark:to-pink-950/20 rounded-xl border border-fuchsia-200/50 dark:border-fuchsia-800/50 flex flex-col items-center justify-center p-6 transition-all duration-300">
                {result ? (
                  <div className="w-full">
                    <WaveformPlayer url={result} />
                    <div className="mt-4 p-3 bg-background/80 rounded-lg">
                      <p className="text-xs text-muted-foreground">Translated output • Ready for download</p>
                    </div>
                  </div>
                ) : isProcessing ? (
                  <div className="flex flex-col items-center justify-center text-center space-y-4">
                    <div className="relative">
                      <div className="w-16 h-16 bg-gradient-to-br from-fuchsia-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg shadow-fuchsia-500/25">
                        <Loader2 className="w-8 h-8 animate-spin text-white" />
                      </div>
                      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-fuchsia-500 to-pink-500 opacity-25 animate-ping"></div>
                    </div>
                    <div className="space-y-2">
                      <p className="text-lg">{processPhase || "Processing..."}</p>
                      {progress > 0 && (
                        <div className="w-48">
                          <Progress value={progress} className="h-2 [&>div]:bg-gradient-to-r [&>div]:from-fuchsia-500 [&>div]:to-pink-500"/>
                          <p className="text-xs text-muted-foreground mt-1 text-center">{progress}% complete</p>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center text-center space-y-4">
                    <PlaceholderWaveform />
                    <div className="space-y-2">
                      <p className="text-muted-foreground">Translation will appear here</p>
                      <p className="text-xs text-muted-foreground">Upload a file and select a language to begin</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Controls */}
        {fileUrl && (
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6 space-y-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-lg flex items-center justify-center">
                  <Zap className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg">Translation Settings</h3>
              </div>
              
              <Select value={targetLanguage} onValueChange={setTargetLanguage} disabled={isProcessing}>
                <SelectTrigger className="w-full h-12 bg-background border-gray-200 dark:border-gray-700 hover:border-fuchsia-300 dark:hover:border-fuchsia-600 transition-colors">
                  <SelectValue placeholder="Select target language" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
                    <SelectItem key={code} value={code} className="h-12">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{flag}</span>
                        <span>{name}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Button 
                className="w-full h-12 bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 text-white shadow-lg shadow-fuchsia-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-fuchsia-500/30" 
                disabled={!file || !targetLanguage || isProcessing} 
                onClick={handleTranslate}
              >
                {isProcessing ? (
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>{processPhase || "Processing..."}</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <AudioWaveform className="w-5 h-5" />
                    <span>Translate Audio</span>
                  </div>
                )}
              </Button>
            </div>
          </Card>
        )}

        {/* Transcripts */}
        {resultTranscripts.source && (
          <Card className="mt-8 border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 rounded-lg flex items-center justify-center">
                  <FileText className="w-4 h-4 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-lg">Transcripts</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium mb-3 text-muted-foreground">Original</h4>
                  <Card className="max-h-40 overflow-y-auto p-4 bg-gray-50 dark:bg-gray-900/50 border-gray-200 dark:border-gray-700">
                    <p className="text-sm leading-relaxed">{resultTranscripts.source}</p>
                  </Card>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-3 text-muted-foreground">
                    Translated ({LANGUAGES[targetLanguage]?.name})
                  </h4>
                  <Card className="max-h-40 overflow-y-auto p-4 bg-fuchsia-50 dark:bg-fuchsia-950/20 border-fuchsia-200 dark:border-fuchsia-800">
                    <p className="text-sm leading-relaxed">{resultTranscripts.target}</p>
                  </Card>
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );

  const renderVideoInterface = () => (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-violet-50/10 dark:to-violet-950/5 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
            <div className="flex items-center space-x-4 mb-4 md:mb-0">
              <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg shadow-violet-500/25">
                {contentType === 'both' ? (
                  <div className="relative">
                    <Film className="w-6 h-6 text-white" />
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
                      <Zap className="w-2 h-2 text-white" />
                    </div>
                  </div>
                ) : (
                  <Film className="w-6 h-6 text-white" />
                )}
              </div>
              <div>
                <h1 className="text-3xl bg-gradient-to-r from-foreground to-violet-600 bg-clip-text text-transparent">
                  {contentType === 'both' ? 'Video Dubbing & Lip Sync' : 'Video Translation'}
                </h1>
                <p className="text-muted-foreground">
                  {contentType === 'both' 
                    ? 'Complete video transformation with voice-over and lip synchronisation' 
                    : 'Translate the audio track of your video content'}
                </p>
              </div>
            </div>
            <Button variant="outline" onClick={handleReset} className="border-violet-200 dark:border-violet-800 hover:bg-violet-50 dark:hover:bg-violet-950/20">
              Change Type
            </Button>
          </div>
          
          {error && (
            <Alert variant="destructive" className="mb-6 border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-950/20">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        {/* Video Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Original Video */}
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-lg flex items-center justify-center">
                  <Film className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                </div>
                <h3 className="text-lg">Original Video</h3>
              </div>
              
              <div className="w-full bg-gradient-to-br from-gray-50 to-gray-100/50 dark:from-gray-900/50 dark:to-gray-800/50 rounded-xl border border-gray-200/50 dark:border-gray-700/50 aspect-video flex items-center justify-center overflow-hidden transition-all duration-300">
                {fileUrl ? (
                  <video ref={originalMediaRef} src={fileUrl} className="w-full h-full object-contain rounded-xl" controls />
                ) : (
                  <label className="cursor-pointer flex flex-col items-center justify-center p-8 h-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors rounded-xl group">
                    <div className="w-16 h-16 bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8 text-gray-600 dark:text-gray-400" />
                    </div>
                    <span className="text-lg text-gray-600 dark:text-gray-400 mb-2">Upload Video</span>
                    <span className="text-xs text-muted-foreground">MP4, AVI, MOV up to 150MB</span>
                    <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
                  </label>
                )}
              </div>
              {fileUrl && (
                <div className="mt-4 p-3 bg-background/80 rounded-lg">
                  <p className="text-xs text-muted-foreground truncate">{file?.name}</p>
                </div>
              )}
            </div>
          </Card>

          {/* Translated Video */}
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-violet-100 to-purple-100 dark:from-violet-900/30 dark:to-purple-900/30 rounded-lg flex items-center justify-center">
                  <Film className="w-4 h-4 text-violet-600 dark:text-violet-400" />
                </div>
                <h3 className="text-lg">
                  {targetLanguage && LANGUAGES[targetLanguage] 
                    ? `${LANGUAGES[targetLanguage].flag} ${LANGUAGES[targetLanguage].name} Translation` 
                    : 'Translated Video'}
                </h3>
              </div>
              
              <div className="w-full bg-gradient-to-br from-violet-50/50 to-purple-50/50 dark:from-violet-950/20 dark:to-purple-950/20 rounded-xl border border-violet-200/50 dark:border-violet-800/50 aspect-video flex items-center justify-center overflow-hidden transition-all duration-300">
                {result ? (
                  <video ref={translatedMediaRef} src={result} className="w-full h-full object-contain rounded-xl" controls />
                ) : (
                  <div className="flex flex-col items-center justify-center text-center space-y-4 p-8">
                    {isProcessing ? (
                      <>
                        <div className="relative">
                          <div className="w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-500 rounded-full flex items-center justify-center shadow-lg shadow-violet-500/25">
                            <Loader2 className="w-8 h-8 animate-spin text-white" />
                          </div>
                          <div className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 opacity-25 animate-ping"></div>
                        </div>
                        <div className="space-y-2">
                          <p className="text-lg">{processPhase || "Processing..."}</p>
                          {progress > 0 && (
                            <div className="w-48">
                              <Progress value={progress} className="h-2 [&>div]:bg-gradient-to-r [&>div]:from-violet-500 [&>div]:to-purple-500"/>
                              <p className="text-xs text-muted-foreground mt-1 text-center">{progress}% complete</p>
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="space-y-4">
                        <div className="w-16 h-16 bg-gradient-to-br from-violet-200 to-purple-200 dark:from-violet-800 dark:to-purple-800 rounded-2xl flex items-center justify-center">
                          <Film className="w-8 h-8 text-violet-600 dark:text-violet-400" />
                        </div>
                        <div className="space-y-2">
                          <p className="text-muted-foreground">Translated video will appear here</p>
                          <p className="text-xs text-muted-foreground">Upload a video and select settings to begin</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Controls */}
        {fileUrl && (
          <Card className="border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6 space-y-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-lg flex items-center justify-center">
                  <Zap className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg">Translation Settings</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Select value={targetLanguage} onValueChange={setTargetLanguage} disabled={isProcessing}>
                  <SelectTrigger className="w-full h-12 bg-background border-gray-200 dark:border-gray-700 hover:border-violet-300 dark:hover:border-violet-600 transition-colors">
                    <SelectValue placeholder="Select target language" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
                      <SelectItem key={code} value={code} className="h-12">
                        <div className="flex items-center space-x-3">
                          <span className="text-lg">{flag}</span>
                          <span>{name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <div className="flex items-center justify-center space-x-6">
                  {(contentType === 'video' || contentType === 'both') && (
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="lip-sync-toggle"
                        checked={applyLipSync}
                        onChange={(e) => setApplyLipSync(e.target.checked)}
                        disabled={isProcessing}
                        className="form-checkbox h-5 w-5 text-violet-600 rounded border-gray-300 focus:ring-violet-500 cursor-pointer"
                      />
                      <label htmlFor="lip-sync-toggle" className="text-sm font-medium text-foreground whitespace-nowrap flex items-center cursor-pointer">
                        Apply Lip Sync 
                        <Zap size={16} className={`ml-2 ${applyLipSync ? "text-yellow-500" : "text-muted-foreground"}`}/>
                      </label>
                    </div>
                  )}
                  {(contentType === 'video' || contentType === 'both') && (
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="voice-cloning-video-toggle"
                        checked={useVoiceCloningVideo}
                        onChange={(e) => setUseVoiceCloningVideo(e.target.checked)}
                        disabled={isProcessing}
                        className="form-checkbox h-5 w-5 text-violet-600 rounded border-gray-300 focus:ring-violet-500 cursor-pointer"
                      />
                      <label htmlFor="voice-cloning-video-toggle" className="text-sm font-medium text-foreground whitespace-nowrap flex items-center cursor-pointer">
                        Preserve Voice 
                        <AudioWaveform size={16} className={`ml-2 ${useVoiceCloningVideo ? "text-violet-500" : "text-muted-foreground"}`} />
                      </label>
                    </div>
                  )}
                </div>
              </div>
              
              <Button
                className="w-full h-12 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white shadow-lg shadow-violet-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-violet-500/30"
                disabled={!file || !targetLanguage || isProcessing}
                onClick={handleTranslate}
              >
                {isProcessing ? (
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>{processPhase || "Processing..."}</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <Film className="w-5 h-5" />
                    <span>
                      {(contentType === 'both' && applyLipSync)
                        ? 'Translate, Dub & Lip Sync Video'
                        : (contentType === 'video' && applyLipSync)
                          ? 'Translate Audio & Lip Sync'
                          : 'Translate Video Audio'
                      }
                    </span>
                  </div>
                )}
              </Button>
            </div>
          </Card>
        )}

        {/* Transcripts */}
        {resultTranscripts.source && (
          <Card className="mt-8 border-0 bg-background/60 backdrop-blur-sm shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-8 h-8 bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 rounded-lg flex items-center justify-center">
                  <FileText className="w-4 h-4 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-lg">Transcripts</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium mb-3 text-muted-foreground">Original</h4>
                  <Card className="max-h-40 overflow-y-auto p-4 bg-gray-50 dark:bg-gray-900/50 border-gray-200 dark:border-gray-700">
                    <p className="text-sm leading-relaxed">{resultTranscripts.source}</p>
                  </Card>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-3 text-muted-foreground">
                    Translated ({LANGUAGES[targetLanguage]?.name})
                  </h4>
                  <Card className="max-h-40 overflow-y-auto p-4 bg-violet-50 dark:bg-violet-950/20 border-violet-200 dark:border-violet-800">
                    <p className="text-sm leading-relaxed">{resultTranscripts.target}</p>
                  </Card>
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen">
      {currentScreen === 'selection' ? renderSelectionScreen() :
       (contentType === 'audio' ? renderAudioInterface() : renderVideoInterface())}
    </div>
  );
};

export default ContentTranslator;