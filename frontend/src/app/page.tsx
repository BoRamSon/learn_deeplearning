'use client'

import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(response.data)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('업로드 실패')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '20px',
      textAlign: 'center'
    }}>
      <h1 style={{ color: '#333', marginBottom: '30px' }}>
        🏭 안전사고 감지 시스템
      </h1>

      {/* 파일 업로드 */}
      <div style={{ 
        border: '2px dashed #ccc', 
        padding: '40px', 
        marginBottom: '20px',
        borderRadius: '10px'
      }}>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ marginBottom: '20px' }}
        />
        
        {file && (
          <div style={{ marginBottom: '20px' }}>
            <p>선택된 파일: {file.name}</p>
            <video 
              src={URL.createObjectURL(file)} 
              controls 
              style={{ maxWidth: '100%', maxHeight: '300px' }}
            />
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            padding: '10px 20px',
            backgroundColor: loading ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {loading ? '분석 중...' : '🔍 분석 시작'}
        </button>
      </div>

      {/* 결과 표시 */}
      {result && (
        <div style={{
          backgroundColor: result.prediction.is_accident ? '#ffe6e6' : '#e6ffe6',
          padding: '20px',
          borderRadius: '10px',
          border: `2px solid ${result.prediction.is_accident ? '#ff4444' : '#44ff44'}`
        }}>
          <h2 style={{ 
            color: result.prediction.is_accident ? '#cc0000' : '#008800',
            marginBottom: '15px'
          }}>
            {result.prediction.is_accident ? '⚠️ 사고 감지!' : '✅ 정상 상황'}
          </h2>
          
          <div style={{ fontSize: '18px', marginBottom: '15px' }}>
            <strong>분류:</strong> {result.prediction.class_name_kr}
          </div>
          
          <div style={{ fontSize: '16px', marginBottom: '20px' }}>
            <strong>신뢰도:</strong> {(result.prediction.confidence * 100).toFixed(1)}%
          </div>

          {/* 확률 분포 */}
          <div style={{ textAlign: 'left' }}>
            <h3>클래스별 확률:</h3>
            {Object.entries(result.probabilities).map(([className, prob]: [string, any]) => (
              <div key={className} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                marginBottom: '5px',
                padding: '5px',
                backgroundColor: 'rgba(255,255,255,0.5)',
                borderRadius: '3px'
              }}>
                <span>{className}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
