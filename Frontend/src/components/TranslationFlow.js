import React, { useState, useRef, useCallback } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Upload, Globe, Play, CheckCircle } from 'lucide-react';

const TranslationFlow = () => {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [selectedLanguages, setSelectedLanguages] = useState([]);
  const [translations, setTranslations] = useState({});
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);

  const languages = [
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'it', name: 'Italian', flag: '🇮🇹' },
    { code: 'pt', name: 'Portuguese', flag: '🇵🇹' }
  ];

  const handleFileUpload = useCallback((event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStep(2);
    }
  }, []);

  const handleLanguageToggle = useCallback((code) => {
    setSelectedLanguages(prev => {
      if (prev.includes(code)) {
        return prev.filter(c => c !== code);
      }
      return [...prev, code];
    });
  }, []);

  const handleContinue = useCallback(() => {
    if (selectedLanguages.length > 0) {
      setStep(3);
      // Start translations here
      setProcessing(true);
      // Mock progress for now
      let progress = 0;
      const interval = setInterval(() => {
        progress += 5;
        setProgress(progress);
        if (progress >= 100) {
          clearInterval(interval);
          setProcessing(false);
        }
      }, 500);
    }
  }, [selectedLanguages]);

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {[1, 2, 3].map((number) => (
          <div key={number} className="flex items-center flex-1">
            <div className={`
              w-8 h-8 rounded-full flex items-center justify-center
              ${step > number ? 'bg-green-500 text-white' : 
                step === number ? 'bg-purple-600 text-white' : 
                'bg-gray-200 text-gray-600'}
            `}>
              {step > number ? <CheckCircle className="w-5 h-5" /> : number}
            </div>
            {number < 3 && (
              <div className={`flex-1 h-1 mx-2 ${
                step > number ? 'bg-green-500' : 'bg-gray-200'
              }`} />
            )}
          </div>
        ))}
      </div>

      <Card className="bg-white shadow-lg">
        <CardContent className="p-6">
          {/* Step 1: Upload */}
          {step === 1 && (
            <div className="text-center">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-12 cursor-pointer hover:border-purple-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-purple-600" />
                <h2 className="text-xl font-semibold mb-2">Upload Your Content</h2>
                <p className="text-gray-600 mb-4">Drag and drop your audio or video file here</p>
                <Button className="bg-purple-600 hover:bg-purple-700">
                  Select File
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileUpload}
                  accept="audio/*,video/*"
                />
              </div>
            </div>
          )}

          {/* Step 2: Language Selection */}
          {step === 2 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Choose Languages</h2>
              <div className="space-y-3 mb-6">
                {languages.map((language) => (
                  <label
                    key={language.code}
                    className="flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      className="w-5 h-5 text-purple-600"
                      checked={selectedLanguages.includes(language.code)}
                      onChange={() => handleLanguageToggle(language.code)}
                    />
                    <Globe className="w-5 h-5 ml-3 mr-2" />
                    <span>{language.flag} {language.name}</span>
                  </label>
                ))}
              </div>
              <Button
                onClick={handleContinue}
                disabled={selectedLanguages.length === 0}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300"
              >
                Continue
              </Button>
            </div>
          )}

          {/* Step 3: Preview Translations */}
          {step === 3 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Preview Translations</h2>
              
              {/* Original Audio */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">Original</h3>
                  <Button size="sm" variant="ghost">
                    <Play className="w-4 h-4" />
                  </Button>
                </div>
                <div className="bg-gray-100 h-12 rounded-lg"></div>
              </div>

              {/* Translated Audio */}
              {selectedLanguages.map(code => {
                const language = languages.find(l => l.code === code);
                return (
                  <div key={code} className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{language?.name}</h3>
                      <Button size="sm" variant="ghost">
                        <Play className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="bg-gray-100 h-12 rounded-lg"></div>
                  </div>
                );
              })}

              <Button 
                className="w-full bg-purple-600 hover:bg-purple-700 mt-4"
                onClick={() => setStep(4)}
              >
                Continue to Export
              </Button>
            </div>
          )}

          {/* Processing Indicator */}
          {processing && (
            <div className="mt-4">
              <Progress value={progress} className="mb-2" />
              <p className="text-sm text-gray-600 text-center">
                Processing translations...
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TranslationFlow;