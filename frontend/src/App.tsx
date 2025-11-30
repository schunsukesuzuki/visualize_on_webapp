import { BrowserRouter, Routes, Route, Navigate} from 'react-router-dom';
import Preprocess from './preprocess';
import Graphicallasso from './graphicallasso';
import Lkj from './lkj';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<div>top</div>} />
        <Route path="/preprocess" element={<Preprocess />} />
        <Route path="/graphical-lasso" element={<Graphicallasso />} />
        <Route path="/lkj" element={<Lkj />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
