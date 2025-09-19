import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: '안전사고 감지',
  description: 'AI 기반 안전사고 감지 시스템',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body style={{ margin: 0, fontFamily: 'Arial, sans-serif' }}>
        {children}
      </body>
    </html>
  )
}
