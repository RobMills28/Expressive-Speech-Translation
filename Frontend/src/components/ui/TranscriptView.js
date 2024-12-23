import React, { useState, useEffect } from 'react';
import { Button } from './button';
import { Card, CardContent } from './card';
import { FileText } from 'lucide-react';

const TranscriptView = ({ sourceText, targetText, targetLang }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasTranscript, setHasTranscript] = useState(false);

  useEffect(() => {
    // Check if we have valid transcript data
    setHasTranscript(Boolean(sourceText || targetText));
  }, [sourceText, targetText]);

  const getLanguageName = (code) => {
    const languageMap = {
      'fra': 'French',
      'deu': 'German',
      'ita': 'Italian',
      'por': 'Portuguese',
      'spa': 'Spanish'
    };
    return languageMap[code] || code;
  };

  if (!hasTranscript) return null;

  return (
    <div className="w-full space-y-4">
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          w-full 
          flex 
          items-center 
          justify-center 
          gap-2
          text-white 
          transition-all 
          duration-200 
          ${isOpen
            ? 'bg-fuchsia-600 hover:bg-fuchsia-700'
            : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700'
          }
        `}
        type="button"
      >
        <FileText className="h-4 w-4" />
        <span>{isOpen ? 'Hide Transcript' : 'Show Transcript'}</span>
      </Button>

      {isOpen && (
        <Card className="mt-4 bg-white/95 shadow-lg">
          <CardContent className="p-6 space-y-6">
            <div>
              <h3 className="font-semibold text-fuchsia-800 mb-2">
                Source Text (English)
              </h3>
              <div className="p-4 rounded-md bg-fuchsia-50 text-gray-700 min-h-[50px] whitespace-pre-wrap">
                {sourceText || 'No source text available'}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-fuchsia-800 mb-2">
                Target Text ({getLanguageName(targetLang)})
              </h3>
              <div className="p-4 rounded-md bg-fuchsia-50 text-gray-700 min-h-[50px] whitespace-pre-wrap">
                {targetText || 'No target text available'}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default TranscriptView;