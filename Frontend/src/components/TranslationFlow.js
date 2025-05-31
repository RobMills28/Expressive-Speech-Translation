// src/components/TranslationFlow.js
import React, { useState, useRef, useEffect } from 'react';
import { Card /* CardContent was unused */ } from "./ui/card"; // Assuming Card is used, CardContent removed
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, Upload, Loader2, Film, /* Mic was unused */ AudioWaveform, Zap } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";
import { Label } from "./ui/label"; // Label can still be used with HTML checkbox

const LANGUAGES = {
  'eng': { name: 'English', flag: '🇺🇸' },
  'fra': { name: 'French', flag: '🇫🇷' },
  'spa': { name: 'Spanish', flag: '🇪🇸' },
  'deu': { name: 'German', flag: '🇩🇪' },
  'ita': { name: 'Italian', flag: '🇮🇹' },
  'por': { name: 'Portuguese', flag: '🇵🇹' },
  'rus': { name: 'Russian', flag: '🇷🇺' },
  'jpn': { name: 'Japanese', flag: '🇯🇵' },
  'cmn': { name: 'Chinese (Simplified)', flag: '🇨🇳' },
  'ukr': { name: 'Ukrainian', flag: '🇺🇦' }
  // Add other languages from VideoSyncInterface if needed for consistency
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
  const [useVoiceCloningVideo, setUseVoiceCloningVideo] = useState(true); // Added state for voice cloning for video

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
    setUseVoiceCloningVideo(true); // Reset voice cloning for video
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    let valid = false;
    const isAudio = uploadedFile.type.startsWith('audio/');
    const isVideo = uploadedFile.type.startsWith('video/');

    if (contentType === 'audio' && isAudio) valid = true;
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
    formData.append('backend', 'cascaded'); // Currently hardcoded backend

    if (contentType === 'video' || contentType === 'both') {
      formData.append('apply_lip_sync', applyLipSync ? 'true' : 'false');
      // Add use_voice_cloning for video types
      formData.append('use_voice_cloning', useVoiceCloningVideo ? 'true' : 'false');
      console.log("Frontend sending apply_lip_sync:", applyLipSync, "use_voice_cloning:", useVoiceCloningVideo);
    }

    const endpoint = contentType === 'audio' ? 'http://localhost:5001/translate' : 'http://localhost:5001/process-video';

    try {
      const response = await fetch(endpoint, { method: 'POST', body: formData });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
        throw new Error(errData.error || `Request failed: ${response.statusText}`);
      }

      if (contentType === 'audio') {
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
        setProgress(100); setProcessPhase('Completed!');
      } else { // Video processing with SSE
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
                    setError(data.error + (data.phase ? ` (during ${data.phase})` : ''));
                    setProcessPhase(`Error: ${data.phase || 'processing'}`);
                    setIsProcessing(false);
                    return;
                }
                if (data.progress !== undefined) setProgress(data.progress);
                if (data.phase) setProcessPhase(data.phase);
                if (data.result) { // Base64 video data
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
                  if (!error) setError("Error processing video stream from server.");
              }
            }
          }
        }
      }
    } catch (err) {
        setError(err.message);
        console.error("Translate error:", err);
        setProcessPhase('Failed.');
    }
    // Only set isProcessing to false if it wasn't already set by an error condition in SSE
    if (isProcessing) { // Check if still true before setting
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

  const renderSelectionScreen = () => (
    <div className="bg-white rounded-lg shadow-sm p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6 text-center">Choose Content Type</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
             onClick={() => handleContentTypeSelect('audio')}>
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <AudioWaveform className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Audio Translation</h3>
          <p className="text-sm text-gray-500 text-center">Podcasts, voice notes, etc.</p>
        </div>
        <div className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
             onClick={() => handleContentTypeSelect('video')}>
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <Film className="h-8 w-8 text-fuchsia-600" />
          </div>
          <h3 className="font-medium mb-2">Video Translation</h3>
          <p className="text-sm text-gray-500 text-center">Translate video audio track.</p>
        </div>
        <div className="flex flex-col items-center p-6 rounded-lg border hover:border-fuchsia-300 cursor-pointer transition-all hover:shadow-md"
             onClick={() => handleContentTypeSelect('both')}>
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center mb-4">
            <div className="relative"><Film className="h-8 w-8 text-fuchsia-600" /><Zap className="h-5 w-5 text-yellow-500 absolute -bottom-1 -right-1" /></div>
          </div>
          <h3 className="font-medium mb-2">Video Dubbing & Lip Sync</h3>
          <p className="text-sm text-gray-500 text-center">Full video voice-over with lip sync.</p>
        </div>
      </div>
    </div>
  );

  const renderAudioInterface = () => (
    // ... (Audio interface remains the same, no voice cloning option here)
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <div className="p-6 border-b flex justify-between items-center">
        <div><h1 className="text-2xl font-semibold">Audio Translation</h1><p className="text-gray-500">Translate speech in audio files</p></div>
        <Button variant="outline" onClick={handleReset}>Change Type</Button>
      </div>
      {error && <div className="px-6 pt-6"><Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert></div>}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        <div>
          <h3 className="font-medium text-lg mb-3">Original Audio</h3>
          <div className="min-h-[200px] bg-gray-100 rounded-lg mb-3 p-4">
            {fileUrl ? (<><audio ref={originalMediaRef} src={fileUrl} className="w-full" controls /><p className="text-xs text-gray-500 mt-2 truncate">{file?.name}</p></>)
             : (<label className="cursor-pointer flex flex-col items-center justify-center h-full min-h-[150px]"><Upload className="w-10 h-10 text-fuchsia-600" /><span className="text-gray-600 mt-2">Upload Audio</span><input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} /></label>)}
          </div>
        </div>
        <div>
          <h3 className="font-medium text-lg mb-3">{targetLanguage && LANGUAGES[targetLanguage] ? `${LANGUAGES[targetLanguage].name} Translation` : 'Translated Audio'}</h3>
          <div className="min-h-[200px] bg-gray-100 rounded-lg p-4">
            {result ? (<><audio ref={translatedMediaRef} src={result} className="w-full" controls /><p className="text-xs text-gray-500 mt-2">Translated output</p></>)
             : (<div className="flex items-center justify-center h-full min-h-[150px] text-gray-500">{isProcessing ? <><Loader2 className="w-8 h-8 animate-spin text-fuchsia-600 mr-2" /><div><p>{processPhase || "Processing..."}</p>{progress > 0 && <Progress value={progress} className="w-32 mt-1 [&>div]:bg-fuchsia-600"/>}</div></> : "Translation will appear here"}</div>)}
          </div>
        </div>
      </div>
      {fileUrl && (
        <div className="p-6 border-t space-y-4">
          <Select value={targetLanguage} onValueChange={setTargetLanguage} disabled={isProcessing}>
            <SelectTrigger className="w-full"><SelectValue placeholder="Select target language" /></SelectTrigger>
            <SelectContent>{Object.entries(LANGUAGES).map(([code, { name, flag }]) => (<SelectItem key={code} value={code}>{`${flag} ${name}`}</SelectItem>))}</SelectContent>
          </Select>
          <Button className="w-full bg-fuchsia-600 hover:bg-fuchsia-700" disabled={!file || !targetLanguage || isProcessing} onClick={handleTranslate}>
            {isProcessing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />{processPhase || "Processing..."}</> : 'Translate Audio'}
          </Button>
        </div>
      )}
      {resultTranscripts.source && (
        <div className="p-6 border-t">
          <h3 className="font-medium text-lg mb-3">Transcripts</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div><h4 className="font-semibold mb-1">Original</h4><Card className="max-h-32 overflow-y-auto p-2 text-sm bg-gray-50">{resultTranscripts.source}</Card></div>
            <div><h4 className="font-semibold mb-1">Translated ({LANGUAGES[targetLanguage]?.name})</h4><Card className="max-h-32 overflow-y-auto p-2 text-sm bg-gray-50">{resultTranscripts.target}</Card></div>
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
                {contentType === 'both' ? 'Video Dubbing & Lip Sync' : 'Video Translation'}
            </h1>
            <p className="text-gray-500">
                {contentType === 'both' ? 'Translate, voice-over, and lip-sync your video' : 'Translate the audio track of your video'}
            </p>
        </div>
        <Button variant="outline" onClick={handleReset}>Change Type</Button>
      </div>
      {error && <div className="px-6 pt-6"><Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert></div>}
      <div className="p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-lg mb-2">Original Video</h3>
            <div className="w-full bg-gray-100 rounded-lg aspect-video max-h-[calc(50vh-8rem)] flex items-center justify-center overflow-hidden">
              {fileUrl ? (<video ref={originalMediaRef} src={fileUrl} className="w-full h-full object-contain" controls />)
               : (<label className="cursor-pointer flex flex-col items-center justify-center p-8 h-full"><Upload className="w-12 h-12 text-fuchsia-600" /><span className="text-gray-600 mt-2">Upload Video</span><input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} /></label>)}
            </div>
             {fileUrl && <p className="text-xs text-gray-500 mt-1 truncate">{file?.name}</p>}
          </div>
          <div>
            <h3 className="font-medium text-lg mb-2">{targetLanguage && LANGUAGES[targetLanguage] ? `${LANGUAGES[targetLanguage].name} Translation` : 'Translated Video'}</h3>
            <div className="w-full bg-gray-100 rounded-lg aspect-video max-h-[calc(50vh-8rem)] flex items-center justify-center overflow-hidden">
              {result ? (<video ref={translatedMediaRef} src={result} className="w-full h-full object-contain" controls />)
               : (<div className="flex flex-col items-center justify-center text-gray-500 p-8 h-full">{isProcessing ? <><Loader2 className="w-10 h-10 animate-spin text-fuchsia-600 mb-3" /><div><p className="text-center">{processPhase || "Processing..."}</p>{progress > 0 && <Progress value={progress} className="w-full mt-2 [&>div]:bg-fuchsia-600"/>}</div></> : "Translated video appears here"}</div>)}
            </div>
          </div>
        </div>
        {fileUrl && (
          <div className="space-y-4 pt-4 border-t mt-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
              <Select value={targetLanguage} onValueChange={setTargetLanguage} disabled={isProcessing}>
                <SelectTrigger className="w-full"><SelectValue placeholder="Select target language" /></SelectTrigger>
                <SelectContent>{Object.entries(LANGUAGES).map(([code, { name, flag }]) => (<SelectItem key={code} value={code}>{`${flag} ${name}`}</SelectItem>))}</SelectContent>
              </Select>

              <div className="flex items-center space-x-4 justify-self-start md:justify-self-end py-2">
                {(contentType === 'video' || contentType === 'both') && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="lip-sync-toggle"
                      checked={applyLipSync}
                      onChange={(e) => setApplyLipSync(e.target.checked)}
                      disabled={isProcessing}
                      className="form-checkbox h-5 w-5 text-fuchsia-600 rounded border-gray-300 focus:ring-fuchsia-500 cursor-pointer"
                    />
                    <label htmlFor="lip-sync-toggle" className="text-sm font-medium text-gray-700 whitespace-nowrap flex items-center cursor-pointer">
                      Apply Lip Sync <Zap size={16} className={`ml-1 ${applyLipSync ? "text-yellow-500" : "text-gray-400"}`}/>
                    </label>
                  </div>
                )}
                 {/* Voice Cloning Toggle for Video */}
                {(contentType === 'video' || contentType === 'both') && (
                    <div className="flex items-center space-x-2">
                        <input
                            type="checkbox"
                            id="voice-cloning-video-toggle"
                            checked={useVoiceCloningVideo}
                            onChange={(e) => setUseVoiceCloningVideo(e.target.checked)}
                            disabled={isProcessing}
                            className="form-checkbox h-5 w-5 text-fuchsia-600 rounded border-gray-300 focus:ring-fuchsia-500 cursor-pointer"
                        />
                        <label htmlFor="voice-cloning-video-toggle" className="text-sm font-medium text-gray-700 whitespace-nowrap flex items-center cursor-pointer">
                            Preserve Voice <AudioWaveform size={16} className={`ml-1 ${useVoiceCloningVideo ? "text-fuchsia-500" : "text-gray-400"}`} />
                        </label>
                    </div>
                )}
              </div>
            </div>
            <Button
              className="w-full bg-fuchsia-600 hover:bg-fuchsia-700 text-white"
              disabled={!file || !targetLanguage || isProcessing}
              onClick={handleTranslate}
            >
              {isProcessing
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />{processPhase || "Processing..."}</>
                : (contentType === 'both' && applyLipSync)
                  ? 'Translate, Dub & Lip Sync Video'
                  : (contentType === 'video' && applyLipSync)
                    ? 'Translate Audio & Lip Sync'
                    : 'Translate Video Audio'
              }
            </Button>
          </div>
        )}
        {resultTranscripts.source && (
          <div className="pt-6 border-t">
            <h3 className="font-medium text-lg mb-3">Transcripts</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div><h4 className="font-semibold mb-1">Original</h4><Card className="max-h-32 overflow-y-auto p-2 text-sm bg-gray-50">{resultTranscripts.source}</Card></div>
              <div><h4 className="font-semibold mb-1">Translated ({LANGUAGES[targetLanguage]?.name})</h4><Card className="max-h-32 overflow-y-auto p-2 text-sm bg-gray-50">{resultTranscripts.target}</Card></div>
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