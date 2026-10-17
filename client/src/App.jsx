import WestCoastMap from "@/components/WestCoastMap";
import Menu from "@/components/Menu";

function App() {
    return (
        <div className="relative">
            <WestCoastMap />
            <div className="absolute left-4 top-4 z-10 backdrop-blur shadow-md bg-background/90 w-64">
                <Menu />
            </div>
        </div>
    );
}

export default App;
