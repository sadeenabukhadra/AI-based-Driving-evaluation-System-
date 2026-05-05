"use client";

import { useState } from "react";
import { questions } from "../data/questions";

export default function Quiz() {
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [done, setDone] = useState(false);

  const current = questions[index];

  const answer = (i: number) => {
    if (i === current.correct) {
      setScore(score + 1);
    }

    if (index + 1 < questions.length) {
      setIndex(index + 1);
    } else {
      setDone(true);
    }
  };

  if (done) {
    return (
      <div className="result">
        <h2>🎯 النتيجة: {score} / {questions.length}</h2>
      </div>
    );
  }

  return (
    <div className="quiz">
      <h3>سؤال {index + 1}</h3>

      {current.image && (
        <img src={current.image} alt="question" />
      )}

      <h2>{current.q}</h2>

      {current.a.map((ans, i) => (
        <button key={i} onClick={() => answer(i)}>
          {ans}
        </button>
      ))}
    </div>
  );
}