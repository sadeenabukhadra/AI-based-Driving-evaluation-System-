import Link from "next/link";

export default function Home() {
  return (
    <main className="home">
      <h1>🚗 امتحان السواقة الأردني</h1>

      <Link href="/quiz">
        <button>ابدأ الامتحان</button>
      </Link>
    </main>
  );
}
