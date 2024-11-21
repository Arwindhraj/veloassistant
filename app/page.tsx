import Link from "next/link";

export default function Home() {
  return (
    <main>
      
      <header className="bg-black-100 shadow-md fixed top-0 left-0 w-full z-50">
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-primary">
                <Link href="/">Velo Assistant</Link>
              </h1>
            </div>

            <nav className="flex items-center space-x-6">
              <button type="button">Home</button>
              <a href="#issue-section" className="hover:text-primary transition">
                Services
              </a>
              
            </nav>
          </div>
        </header>

        <section>
        <div className="relative w-full h-screen">
        <canvas id="myCanvas" className="absolute inset-0 w-full h-full" />
        </div>
        </section>

        <div className="container mx-auto px-4 py-12">
        <section id="issue-section" className="text-center mb-16">
          <h1 className="text-4xl font-bold mb-4">Velo Assistant</h1>
          <p className="text-xl text-muted-foreground mb-4">
            Your intelligent companion for all vehicle-related information
          </p>
        </section>

        <section className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <Link href='/issue'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Issue</h2>
              <p className="text-muted-foreground text-center">
                Explore your Issue
              </p>
            </div>
          </div>
          </Link>

          <Link href='/spare'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Spare</h2>
              <p className="text-muted-foreground text-center">
                Explore your Spare
              </p>
            </div>
          </div>
          </Link>

          <Link href='/vechicleinfo'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Vehicle Information</h2>
              <p className="text-muted-foreground text-center">
                Explore your Vehicle Information
              </p>
            </div>
          </div>
          </Link>

          <Link href='/issue'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Issue</h2>
              <p className="text-muted-foreground text-center">
                Explore your Issue
              </p>
            </div>
          </div>
          </Link>

          <Link href='/spare'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Spare</h2>
              <p className="text-muted-foreground text-center">
                Explore your Spare
              </p>
            </div>
          </div>
          </Link>

          <Link href='/vechicleinfo'>
          <div className="group relative overflow-hidden rounded-lg border p-8 hover:border-primary transition-colors">
            <div className="flex flex-col items-center">
              <h2 className="text-2xl font-semibold mb-2">Vehicle Information</h2>
              <p className="text-muted-foreground text-center">
                Explore your Vehicle Information
              </p>
            </div>
          </div>
          </Link>
        
        </section>
        <section className="bg-gray-800 text-white py-12 mt-16">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-xl font-bold mb-4">About Us</h3>
                <p className="text-sm">
                  We are committed to providing exceptional services and solutions for your vehicle needs. Our mission is to make vehicle care simpler and more accessible.
                </p>
              </div>

              <div>
                <h3 className="text-xl font-bold mb-4">Contact Us</h3>
                <ul className="space-y-2">
                  <li>
                    <span className="font-semibold">Email:</span> zympacttech@gmail.com
                  </li>
                  <li>
                    <span className="font-semibold">Phone:</span> +91 80983 91340
                  </li>
                  <li>
                    <span className="font-semibold">Address:</span> 56/A Ground Floor, C.S.I Malayalam Church, Rathinapuri, Coimbatore, Tamil Nadu, India
                  </li>
                </ul>
              </div>

              <div className="text-center md:text-right">
                <h3 className="text-xl font-bold mb-4">Credits</h3>
                <p className="text-sm">
                  Designed and developed by <span className="font-semibold">Zympact Technology</span>.
                </p>
                <p className="text-sm mt-2">
                  © {new Date().getFullYear()} Zympact Technology Pvt. Ltd. All rights reserved.
                </p>
              </div>
            </div>

            <div className="text-center mt-8">
              <p className="text-sm">
                <span>Powered by </span>
                <a href="https://www.example.com" className="text-primary hover:underline">
                Zympact Technology
                </a>
              </p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
