// src/components/ui/select.js
import React from 'react';

export function Select({ children, onValueChange, value, ...props }) {
  const handleChange = (event) => {
    if (typeof onValueChange === 'function') {
      onValueChange(event.target.value);
    }
  };

  return (
    <select 
      className="w-full p-2 border rounded" 
      onChange={handleChange} 
      value={value}
      {...props}
    >
      {children}
    </select>
  );
}

export function SelectTrigger({ children, ...props }) {
  return <div {...props}>{children}</div>;
}

export function SelectValue({ placeholder }) {
  return <span>{placeholder}</span>;
}

export function SelectContent({ children }) {
  return <>{children}</>;
}

export function SelectItem({ children, value }) {
  return <option value={value}>{children}</option>;
}