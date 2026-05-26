아래는 VWorld **WMTS API 레퍼런스** 페이지 내용을 마크다운으로 정리한 것입니다. ([브이월드][1])

---

# VWorld WMTS API 레퍼런스 정리

## 1. 개요

VWorld는 **WMTS(Web Map Tile Service)** 를 통해 고품질의 배경지도를 제공합니다.

인증된 API Key를 사용하여 요청 URL을 서버로 전송하면 VWorld WMTS 서비스를 사용할 수 있습니다.

---

## 2. WMTS GetTile 요청 URL

```text
https://api.vworld.kr/req/wmts/1.0.0/{key}/{layer}/{tileMatrix}/{tileRow}/{tileCol}.{tileType}
```

### 예시

```text
https://api.vworld.kr/req/wmts/1.0.0/{key}/Base/11/793/1746.png
```

---

## 3. WMTS GetTile 요청 URL: 해외위성영상

```text
https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/themes/{category}/{year}/{city}/{tileMatrix}/{tileRow}/{tileCol}.{tileType}
```

### 예시

```text
https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/themes/cities/2025/Oslo/11/1086/596.png
```

---

## 4. WMTS GetCapabilities 요청 URL

```text
https://api.vworld.kr/req/wmts/1.0.0/{key}/WMTSCapabilities.xml
```

---

## 5. 요청 파라미터

| 파라미터         | 선택    | 설명                | 유효값                                                                                           |
| ------------ | ----- | ----------------- | --------------------------------------------------------------------------------------------- |
| `key`        | `M/1` | 발급받은 api key      | 발급받은 API Key                                                                                  |
| `layer`      | `M/1` | 요청 서비스 버전         | `Base`, `white`, `midnight`, `Hybrid`, `Satellite`                                            |
| `tileMatrix` | `M/1` | 지도 레벨             | `Base : 6~19`<br>`white : 6~18`<br>`midnight : 6~18`<br>`Hybrid : 6~19`<br>`Satellite : 6~19` |
| `tileRow`    | `M/1` | Google Index Y좌표값 | 숫자                                                                                            |
| `tileCol`    | `M/1` | Google Index X좌표값 | 숫자                                                                                            |
| `tileType`   | `M/1` | Tile 확장자          | `Base : png`<br>`white : png`<br>`midnight : png`<br>`Hybrid : png`<br>`Satellite : jpeg`     |

---

## 6. 레이어별 지도 레벨 범위

| 레이어         | `tileMatrix` 범위 |
| ----------- | --------------: |
| `Base`      |        `6 ~ 19` |
| `white`     |        `6 ~ 18` |
| `midnight`  |        `6 ~ 18` |
| `Hybrid`    |        `6 ~ 19` |
| `Satellite` |        `6 ~ 19` |

---

## 7. 레이어별 타일 확장자

| 레이어         | `tileType` |
| ----------- | ---------- |
| `Base`      | `png`      |
| `white`     | `png`      |
| `midnight`  | `png`      |
| `Hybrid`    | `png`      |
| `Satellite` | `jpeg`     |

---

## 8. 기본 사용 예제

```text
https://api.vworld.kr/req/wmts/1.0.0/[KEY]/Base/11/793/1746.png
```

`[KEY]` 부분에는 VWorld에서 발급받은 API Key를 넣으면 됩니다.

---

## 9. 오류 응답 형식

오류가 발생하면 `ExceptionReport` 형태로 오류 정보가 반환됩니다.

| 항목명               | 타입    | 설명                                 |
| ----------------- | ----- | ---------------------------------- |
| `ExceptionReport` | 오류 정보 | 오류 응답의 Root                        |
| `Exception`       | 문자    | 속성으로 `exceptionCode`, `locator` 포함 |
| `ExceptionText`   | 문자    | 오류 메시지                             |

---

## 10. 오류 메시지

| exceptionCode           | ExceptionText       | locator           |
| ----------------------- | ------------------- | ----------------- |
| `FileNotFound`          | 파일을 찾지 못했습니다.       | -                 |
| `MissingParameterValue` | 필수 요청변수 값이 누락되었습니다. | 누락된 요청변수 명        |
| `InvalidParameterValue` | 요청변수 값이 유효하지 않습니다.  | 유효하지 않은 값의 요청변수 명 |
| `NoApplicableCode`      | 서버 오류입니다.           | -                 |

---

## 11. 참고

OGC WMTS 표준 문서도 함께 참고할 수 있습니다.

```text
https://www.ogc.org/standard/wmts
```

[1]: https://www.vworld.kr/dev/v4dv_wmtsguide_s001.do "브이월드"
