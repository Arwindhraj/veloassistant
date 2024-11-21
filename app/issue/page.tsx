"use client";

import { useState } from "react";
import Link from "next/link";

// Define initial data
const data = {
  vehicleTypes: ["2-wheeler", "4-wheeler"],
  brands: {
    "2-wheeler": ["Honda", "Yamaha", "Suzuki"],
    "4-wheeler": ["Toyota", "Ford", "Tesla"],
  },
  fuelTypes: ["Petrol", "Diesel", "EV"],
  models: {
    Honda: ["Activa", "Shine", "Unicorn"],
    Yamaha: ["FZ", "R15", "MT-15"],
    Suzuki: ["Access", "Gixxer"],
    Toyota: ["Corolla", "Camry", "Yaris"],
    Ford: ["Mustang", "Explorer", "Fiesta"],
    Tesla: ["Model S", "Model X", "Model 3"],
  },
  years: ["2020", "2021", "2022", "2023"],
};

export default function Issue() {
  // State to hold current selection for each dropdown
  const [vehicleType, setVehicleType] = useState("");
  const [brand, setBrand] = useState("");
  const [fuelType, setFuelType] = useState("");
  const [model, setModel] = useState("");
  const [year, setYear] = useState("");
  

  // Handlers for changing each dropdown
  const handleVehicleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setVehicleType(e.target.value);
    setBrand(""); // Reset brand, model, and year when vehicle type changes
    setModel("");
    setYear("");
  };

  const handleBrandChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setBrand(e.target.value);
    setModel(""); // Reset model and year when brand changes
    setYear("");
  };

  const handleFuelTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFuelType(e.target.value);
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setModel(e.target.value);
  };

  const handleYearChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setYear(e.target.value);
  };

  // Function to handle confirmation and saving data
  const handleConfirmDetails = async () => {
    // Validate all fields are selected
    if (!vehicleType || !brand || !fuelType || !model || !year) {
      alert("Please select all fields before confirming");
      return;
    }

    // Temporary User ID
    const userId = "1010101010101";
    const objectId = "20202020220";

    // const detail_user = [
    //   userId,
    //   [objectId, vehicleType, brand, fuelType, model, year],
    // ];

    const newSelection = {
      userId,
      objectId,
      vehicleType,
      brand,
      fuelType,
      model,
      year,
    };

    try {
      // Save to API
      const response = await fetch("/api/selections", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newSelection),
      });

      if (!response.ok) {
        throw new Error("Failed to save data");
      }

      // Update local state
      
      // alert('Details saved successfully!');

      // Optional: Reset form
      setVehicleType("");
      setBrand("");
      setFuelType("");
      setModel("");
      setYear("");
    } catch (error) {
      // alert('Failed to save details. Please try again.');
      console.error("Error saving details:", error);
    }
  };

  return (
    <main className="min-h-screen p-8">
      <section className="max-w-2xl mx-auto mt-16">
        <h1 className="text-3xl font-bold mb-8 text-center">Choose the Vehicle Details</h1>

        <div className="space-y-6">
          <div className="flex flex-col">
            <label className="block text-sm font-medium mb-2">
              Vehicle Type
              <select
                value={vehicleType}
                onChange={handleVehicleTypeChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-black"
              >
                <option value="">Select Vehicle Type</option>
                {data.vehicleTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="flex flex-col">
            <label className="block text-sm font-medium mb-2">
              Brand
              <select
                value={brand}
                onChange={handleBrandChange}
                disabled={!vehicleType}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed text-black"
              >
                <option value="">Select Brand</option>
                {vehicleType &&
                data.brands[vehicleType as "2-wheeler" | "4-wheeler"].map((brandOption) => (
                  <option key={brandOption} value={brandOption}>
                    {brandOption}
                  </option>
                ))}

              </select>
            </label>
          </div>

          <div className="flex flex-col">
            <label className="block text-sm font-medium mb-2">
              Fuel Type
              <select
                value={fuelType}
                onChange={handleFuelTypeChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-black"
              >
                <option value="">Select Fuel Type</option>
                {data.fuelTypes.map((fuel) => (
                  <option key={fuel} value={fuel}>
                    {fuel}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="flex flex-col">
            <label className="block text-sm font-medium mb-2">
              Model
              <select
                value={model}
                onChange={handleModelChange}
                disabled={!brand}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed text-black"
              >
                <option value="">Select Model</option>
                {brand &&
                (brand in data.models) &&
                data.models[brand as keyof typeof data.models].map((modelOption) => (
                  <option key={modelOption} value={modelOption}>
                    {modelOption}
                  </option>
                ))}

              </select>
            </label>
          </div>

          <div className="flex flex-col">
            <label className="block text-sm font-medium mb-2">
              Year
              <select
                value={year}
                onChange={handleYearChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-black"
              >
                <option value="">Select Year</option>
                {data.years.map((yearOption) => (
                  <option key={yearOption} value={yearOption}>
                    {yearOption}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <div className="mt-8 text-center">
          <Link
            href="/assistant"
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
          >
            <button onClick={handleConfirmDetails}>Let&apos;s Go..</button>
          </Link>
        </div>
      </section>
    </main>
  );
}
