'use client';

import React from 'react';

interface Lesson {
  id: string;
  title: string;
  description: string;
  action?: () => void;
}

interface LessonSidebarProps {
  activeLesson: string | null;
  onLessonClick: (lessonId: string | null) => void;
  onAction?: (action: () => void) => void;
}

const lessons: Lesson[] = [
  {
    id: 'line',
    title: '1. What is a line? (m, b)',
    description: 'A line is defined by slope (m) and intercept (b). The equation y = mx + b shows how y changes with x.',
  },
  {
    id: 'prediction',
    title: '2. Prediction vs actual',
    description: 'For any x value, the line predicts a y value. This is our model\'s prediction.',
  },
  {
    id: 'residuals',
    title: '3. Residuals',
    description: 'Residuals are the vertical distances from data points to the line. They show prediction errors.',
  },
  {
    id: 'loss',
    title: '4. Loss = average squared residual',
    description: 'Loss (MSE) is the average of squared residuals. We minimize this to find the best line.',
  },
  {
    id: 'auto-fit',
    title: '5. Best fit line (auto)',
    description: 'The normal equation finds the optimal m and b that minimize MSE mathematically.',
  },
  {
    id: 'gradient-descent',
    title: '6. Gradient descent (learning)',
    description: 'Gradient descent iteratively adjusts m and b by following the gradient (slope) of the loss function.',
  },
  {
    id: 'outliers',
    title: '7. Outliers & robust loss',
    description: 'MSE is sensitive to outliers. MAE and Huber loss are more robust alternatives.',
  },
  {
    id: 'limitations',
    title: '8. Limitations (non-linear)',
    description: 'Linear regression only works for linear relationships. Non-linear data requires different models.',
  },
];

export default function LessonSidebar({ activeLesson, onLessonClick }: LessonSidebarProps) {
  return (
    <div className="w-64 flex-shrink-0 bg-white rounded-lg shadow-sm p-4 overflow-y-auto">
      <h2 className="text-lg font-bold text-gray-800 mb-4">Lessons</h2>
      <div className="space-y-2">
        {lessons.map((lesson) => (
          <button
            key={lesson.id}
            onClick={() => {
              onLessonClick(lesson.id === activeLesson ? null : lesson.id);
            }}
            className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
              activeLesson === lesson.id
                ? 'bg-blue-100 text-blue-800 border-2 border-blue-500'
                : 'bg-gray-50 text-gray-700 hover:bg-gray-100 border-2 border-transparent'
            }`}
          >
            <div className="font-semibold mb-1">{lesson.title}</div>
            {activeLesson === lesson.id && (
              <div className="text-xs text-gray-600 mt-1">{lesson.description}</div>
            )}
          </button>
        ))}
      </div>
      {activeLesson && (
        <button
          onClick={() => onLessonClick(null)}
          className="mt-4 w-full px-3 py-2 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300"
        >
          Clear Selection
        </button>
      )}
    </div>
  );
}

