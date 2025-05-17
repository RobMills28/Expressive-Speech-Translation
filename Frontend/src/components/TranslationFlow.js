import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, Upload, Loader2, Play, Pause, Film, Mic, AudioWaveform } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";
// import BackendSelector from './BackendSelector';

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
  const [currentScreen, setCurrentScreen] = useState('selection');
  const [contentType, setContentType] = useState(null);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processPhase, setProcessPhase] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [resultTranscripts, setResultTranscripts] = useState({ source: '', target: '' });
  const [isPlaying, setIsPlaying] = useState(false);
  const [useVoiceCloning, setUseVoiceCloning] = useState(true);
  const [backendType, setBackendType] = useState('cascaded');

  const mediaRef = useRef(null);
  const resultRef = useRef(null);
  const audioRef = useRef(null);
  const resultAudioRef = useRef(null);

  const handleContentTypeSelect = (type) => {
    setContentType(type);
    setCurrentScreen('translator');
    setFile(null);
    setFileUrl(null);
    setResult(null);
    setResultTranscripts({ source: '', target: '' });
    setError('');
    setProgress(0);
    setProcessPhase('');
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    const isAudioType = uploadedFile.type.startsWith('audio/');
    const isVideoType = uploadedFile.type.startsWith('video/');
    let valid = false;

    if (contentType === 'audio' && isAudioType) valid = true;
    if (contentType === 'video' && isVideoType) valid = true;
    if (contentType === 'both' && isVideoType) valid = true; // 'both' implies video input for this component

    if (!valid) {
      setError(`Please upload a valid ${contentType === 'audio' ? 'audio' : 'video'} file.`);
      return;
    }

    if (uploadedFile.size > 150 * 1024 * 1024) {
      setError('File size should be less than 150MB.');
      return;
    }

    setFile(uploadedFile);
    setFileUrl(URL.createObjectURL(uploadedFile));
    setError('');
    setResult(null);
    setResultTranscripts({ source: '', target: '' });
    setProgress(0);
    setProcessPhase('');
  };
  
  useEffect(() => {
    const audioElement = audioRef.current;
    if (audioElement) {
      const onPlay = () => setIsPlaying(true);
      const onPause = () => setIsPlaying(false);
      const onEnded = () => setIsPlaying(false);
      audioElement.addEventListener('play', onPlay);
      audioElement.addEventListener('pause', onPause);
      audioElement.addEventListener('ended', onEnded);
      return () => {
        audioElement.removeEventListener('play', onPlay);
        audioElement.removeEventListener('pause', onPause);
        audioElement.removeEventListener('ended', onEnded);
      };
    }
  }, [fileUrl]);


  const handleTranslate = async () => {
    if (!file) {
      setError('Please upload a file first.');
      return;
    }
    if (!targetLanguage) {
      setError('Please select a target language.');
      return;
    }

    try {
      setIsProcessing(true);
      setProgress(0);
      setProcessPhase('Preparing content...');
      setError('');
      setResult(null);
      setResultTranscripts({ source: '', target: '' });

      const formData = new FormData();
      const fileKey = (contentType === 'audio') ? 'file' : 'video';
      formData.append(fileKey, file);
      formData.append('target_language', targetLanguage);
      formData.append('backend', backendType);
      formData.append('use_voice_cloning', useVoiceCloning ? 'true' : 'false');

      const endpoint = (contentType === 'audio') ?
        'http://localhost:5001/translate' :
        'http://localhost:5001/process-video';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `Request failed with status ${response.status}` }));
        throw new Error(errorData.error || `Failed to process ${contentType}`);
      }
      
      if (contentType === 'audio') {
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        const audioBlob = new Blob(
            [Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))],
            { type: 'audio/wav' }
        );
        const audioUrl = URL.createObjectURL(audioBlob);
        setResult(audioUrl);
        setResultTranscripts(data.transcripts || { source: '', target: '' });
        setProgress(100);
        setProcessPhase('Translation complete!');

      } else { 
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || '';

          for (const message of messages) {
            if (message.trim().startsWith('data: ')) {
              try {
                const jsonStr = message.trim().slice(6);
                const data = JSON.parse(jsonStr);

                if (data.error) throw new Error(data.error);
                if (data.progress !== undefined) setProgress(data.progress);
                if (data.phase) setProcessPhase(data.phase);
                if (data.result) {
                  const videoBlob = new Blob(
                    [Uint8Array.from(atob(data.result), c => c.charCodeAt(0))],
                    { type: 'video/mp4' }
                  );
                  const videoUrl = URL.createObjectURL(videoBlob);
                  setResult(videoUrl);
                }
                if (data.transcripts) {
                    setResultTranscripts(data.transcripts);
                }
              } catch (e) {
                console.error('Error parsing SSE message:', e, message);
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
       if (!result && progress !== 100 && !error) { 
        setProcessPhase(prev => prev.includes('fail') || prev.includes('error') ? prev : 'Processing finished.');
      }
    }
  };

  const handleReset = () => {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    if (result) URL.revokeObjectURL(result);
    setCurrentScreen('selection');
    setContentType(null);
    setFile(null);
    setFileUrl(null);
    setTargetLanguage(null);
    setResult(null);
    setResultTranscripts({ source: '', target: '' });
    setError('');
    setProgress(0);
    setProcessPhase('');
    setIsPlaying(false);
  };

  const renderSelectionScreen = () => (
    <div className="bg-white rounded-lg shadow-sm p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6 text-center">Choose Content Type</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8"> {/* Restored 3 columns */}
        <div
          className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
          onClick={() => handleContentTypeSelect('audio')}
        >
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <AudioWaveform className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Audio Translation</h3>
          <p className="text-sm text-gray-500 text-center">Translate podcasts, voice notes, and other audio content.</p>
        </div>
        <div
          className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
          onClick={() => handleContentTypeSelect('video')}
        >
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <Film className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Video Translation</h3>
          <p className="text-sm text-gray-500 text-center">Translate and synchronize speech in your video files.</p>
        </div>
        <div // Restored "Audio + Video" option
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
          <p className="text-sm text-gray-500 text-center">Comprehensive video translation with voice cloning.</p>
        </div>
      </div>
    </div>
  );

  const renderAudioInterface = () => (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <div className="p-6 border-b flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">Audio Translation</h1>
          <p className="text-gray-500">Translate speech in your audio files</p>
        </div>
        <Button variant="outline" onClick={handleReset}>Change Content Type</Button>
      </div>
      {error && <div className="px-6 pt-6"><Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert></div>}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        <div>
          <h3 className="font-medium text-lg mb-3">Original Audio</h3>
          <div className="min-h-[200px] bg-gray-100 rounded-lg overflow-hidden mb-3 flex flex-col items-center justify-center p-4">
            {fileUrl ? (
              <>
                <audio ref={audioRef} src={fileUrl} className="w-full mb-4" controls />
                 <p className="text-sm text-gray-600 truncate w-full text-center">{file?.name}</p>
              </>
            ) : (
              <label className="cursor-pointer flex flex-col items-center justify-center h-full">
                <Upload className="w-10 h-10 text-fuchsia-600" />
                <span className="text-gray-600 mt-2">Upload Audio</span>
                <input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} />
              </label>
            )}
          </div>
          {fileUrl && !result && (
            <div className="space-y-4">
              <Select value={targetLanguage || ""} onValueChange={setTargetLanguage} disabled={isProcessing}>
                <SelectTrigger className="w-full"><SelectValue placeholder="Select target language" /></SelectTrigger>
                <SelectContent>{Object.entries(LANGUAGES).map(([code, { name, flag }]) => (<SelectItem key={code} value={code}><span className="flex items-center gap-2"><span>{flag}</span><span>{name}</span></span></SelectItem>))}</SelectContent>
              </Select>
              <Button className="w-full bg-fuchsia-600 hover:bg-fuchsia-700" disabled={!file || !targetLanguage || isProcessing} onClick={handleTranslate}>
                {isProcessing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Processing...</> : 'Translate Audio'}
              </Button>
            </div>
          )}
        </div>
        <div>
          <h3 className="font-medium text-lg mb-3">{targetLanguage && LANGUAGES[targetLanguage] ? `${LANGUAGES[targetLanguage].name} Translation` : 'Translated Audio'}</h3>
          <div className="min-h-[200px] bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center p-4">
            {result ? (
               <>
                <audio ref={resultAudioRef} src={result} className="w-full mb-4" controls />
               </>
            ) : (
              <div className="text-center text-gray-500">
                {isProcessing ? <><Loader2 className="w-8 h-8 animate-spin text-fuchsia-600 mb-2 mx-auto" /><p className="mb-1">Processing...</p><p className="text-sm text-gray-400">{processPhase}</p></> : "Translated audio will appear here"}
              </div>
            )}
          </div>
          {isProcessing && <div className="mt-3"><Progress value={progress} className="[&>div]:bg-fuchsia-600" /></div>}
        </div>
      </div>
        {resultTranscripts.source && resultTranscripts.target && (
            <div className="p-6 border-t">
                <h3 className="font-medium text-lg mb-3">Transcripts</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h4 className="font-semibold mb-1 text-gray-700">Original (English)</h4>
                        <Card className="max-h-40 overflow-y-auto p-3 text-sm bg-gray-50 text-gray-600">{resultTranscripts.source}</Card>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-1 text-gray-700">Translated ({LANGUAGES[targetLanguage]?.name || targetLanguage})</h4>
                        <Card className="max-h-40 overflow-y-auto p-3 text-sm bg-gray-50 text-gray-600">{resultTranscripts.target}</Card>
                    </div>
                </div>
            </div>
        )}
    </div>
  );

  const renderVideoInterface = () => (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <div className="p-6 border-b flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">
            {contentType === 'both' ? 'Audio + Video Translation' : 'Video Translation'}
          </h1>
          <p className="text-gray-500">
            {contentType === 'both' ? 'Translate speech and synchronize voice in your videos' : 'Translate and synchronize speech in your videos'}
          </p>
        </div>
        <Button variant="outline" onClick={handleReset}>Change Content Type</Button>
      </div>
      {error && <div className="px-6 pt-6"><Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert></div>}
      
      <div className="p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left side - Original video */}
          <div>
            <h3 className="font-medium text-lg mb-3">Original Video Preview</h3>
            {/* Apply aspect-square and control max height for square-ish look */}
            <div className="w-full bg-gray-100 rounded-lg overflow-hidden aspect-square max-h-[calc(50vh-4rem)] flex items-center justify-center">
              {fileUrl ? (
                <video ref={mediaRef} src={fileUrl} className="w-full h-full object-contain" controls />
              ) : (
                <label className="cursor-pointer flex flex-col items-center justify-center p-8">
                  <Upload className="w-12 h-12 text-fuchsia-600" />
                  <span className="text-gray-600 mt-2">Upload Video</span>
                  <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
                </label>
              )}
            </div>
          </div>
          {/* Right side - Translated video */}
          <div>
            <h3 className="font-medium text-lg mb-3">{targetLanguage && LANGUAGES[targetLanguage] ? `${LANGUAGES[targetLanguage].name} Translation` : 'Translated Result'}</h3>
            <div className="w-full bg-gray-100 rounded-lg overflow-hidden aspect-square max-h-[calc(50vh-4rem)] flex items-center justify-center">
              {result ? (
                <video ref={resultRef} src={result} className="w-full h-full object-contain" controls />
              ) : (
                <div className="flex flex-col items-center justify-center text-gray-500 p-8">
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-10 h-10 animate-spin text-fuchsia-600 mb-3" />
                      <div className="text-center">
                        <p className="mb-1">Processing your video...</p>
                        <p className="text-sm text-gray-400">{processPhase}</p>
                      </div>
                    </>
                  ) : "Translated video will appear here"}
                </div>
              )}
            </div>
          </div>
        </div>

        {fileUrl && (
            <div className="space-y-4 pt-4 border-t mt-2">
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
                    <Select value={targetLanguage || ""} onValueChange={setTargetLanguage} disabled={isProcessing}>
                        <SelectTrigger className="w-full"><SelectValue placeholder="Select target language" /></SelectTrigger>
                        <SelectContent>{Object.entries(LANGUAGES).map(([code, { name, flag }]) => (<SelectItem key={code} value={code}><span className="flex items-center gap-2"><span>{flag}</span><span>{name}</span></span></SelectItem>))}</SelectContent>
                    </Select>
                    <Button className="w-full bg-fuchsia-600 hover:bg-fuchsia-700" disabled={!file || !targetLanguage || isProcessing} onClick={handleTranslate}>
                        {isProcessing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Processing...</> : 'Translate & Synchronize'}
                    </Button>
                </div>
                {isProcessing && (
                    <div className="mt-3">
                        <Progress value={progress} className="[&>div]:bg-fuchsia-600" />
                        <p className="text-center text-sm text-gray-500 mt-1">{processPhase || "Processing..."}</p>
                    </div>
                )}
            </div>
        )}
        {resultTranscripts.source && resultTranscripts.target && (
            <div className="pt-6 border-t">
                <h3 className="font-medium text-lg mb-3">Transcripts</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h4 className="font-semibold mb-1 text-gray-700">Original (English)</h4>
                        <Card className="max-h-40 overflow-y-auto p-3 text-sm bg-gray-50 text-gray-600">{resultTranscripts.source}</Card>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-1 text-gray-700">Translated ({LANGUAGES[targetLanguage]?.name || targetLanguage})</h4>
                        <Card className="max-h-40 overflow-y-auto p-3 text-sm bg-gray-50 text-gray-600">{resultTranscripts.target}</Card>
                    </div>
                </div>
            </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="mx-auto px-4 py-6 max-w-full">
      {currentScreen === 'selection' ? renderSelectionScreen() :
       (contentType === 'audio' ? renderAudioInterface() : renderVideoInterface())}
    </div>
  );
};

export default ContentTranslator;